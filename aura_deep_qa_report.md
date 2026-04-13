
# Aura Deep QA Session - 2026-04-10 21:16:51.118377

## Startup Logs
```
2026-04-10 21:14:21,246 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
🔁 Relaunching Aura with preferred interpreter: /opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/bin/python3.12
2026-04-10 21:14:21,414 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
🖥️ HEADLESS MODE ACTIVATED
🔍 Verifying Environment Integrity...
📍 RUNTIME PATH Diagnostic:
   • __file__: /Users/bryan/.aura/live-source/aura_main.py
   • sys.executable: /opt/homebrew/opt/python@3.12/bin/python3.12
   • sys.path: ['/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages', '/Users/bryan/.aura/live-source', '/Users/bryan/.aura/live-source', '/Users/bryan/.aura/live-source', '/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/lib/python312.zip', '/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/lib/python3.12', '/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload', '/opt/homebrew/lib/python3.12/site-packages']
   • core.__file__: /Users/bryan/.aura/live-source/core/__init__.py
🛠️  Pending patch detected. Validating syntax...
pending_patch.py passed syntax check. Run patch_applicator.py to apply.
🛡️ uvloop disabled for this runtime profile. Set AURA_ENABLE_UVLOOP=1 to force-enable it.
🛡️  REAPER ACTIVE (Survives SIGKILL). Monitoring Kernel PID: 48064
🔒 Instance lock acquired: orchestrator (PID: 48064)
Scheduler substrate initialized.
AuraEventBus initialized (Redis: True).
✅ [EVENT_BUS] Kernel signaling READY.
ThoughtEmitter initialized.
2026-04-10 21:14:21,609 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
[REAPER] Watching Kernel PID 48064
Initializing Modular Service Providers (is_proxy=False)...
🍄 [MYCELIUM] Pathway Hardwired: 'reflex_identity' → identity_reflex (priority=2.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'reflex_status' → status_reflex (priority=2.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'direct_web_search' → search_web (priority=1.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'direct_self_repair' → self_repair (priority=1.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'reflex_help' → help_reflex (priority=2.0, groups=[])
🍄 [MYCELIUM] 🌿 Neural Root ESTABLISHED: voice_presence->hardware:macos_say
🍄 [MYCELIUM] Network Online v4.0 (Hardened) — Enterprise Grade.
🍄 [MYCELIUM] Linking Transcendence Layer: 'meta_cognition' -> MetaEvolutionEngine
🍄 [MYCELIUM] Hypha established: meta_cognition->cognition
🍄 [MYCELIUM] Hypha established: cognition->meta_cognition
🍄 [MYCELIUM] Hypha established: qualia->phenomenology
🍄 [MYCELIUM] Hypha established: consciousness->global_workspace
🍄 [MYCELIUM] Hypha established: sentience->autonomy
🍄 [MYCELIUM] 👁️ Consciousness Hyphae established.
🍄 [MYCELIUM] Hypha established: cognition->llm
🍄 [MYCELIUM] Hypha established: memory->cognition
🍄 [MYCELIUM] 🌿 Neural Root ESTABLISHED: llm->hardware:gpu_metal
✅ All modular services registered and validated (Lock deferred).
🧠 MindTick: Registered phase 'proprioceptive_loop'
🧠 MindTick: Registered phase 'social_context'
🧠 MindTick: Registered phase 'sensory_ingestion'
🧠 MindTick: Registered phase 'memory_retrieval'
🧠 MindTick: Registered phase 'affect_update'
🧠 MindTick: Registered phase 'executive_closure'
🧠 MindTick: Registered phase 'cognitive_routing'
🧠 MindTick: Registered phase 'response_generation'
🧠 MindTick: Registered phase 'memory_consolidation'
🧠 MindTick: Registered phase 'identity_reflection'
🧠 MindTick: Registered phase 'initiative_generation'
🧠 MindTick: Registered phase 'consciousness'
📋 MindTick: TaskRegistry heartbeat wired.
ChaosEngine initialized (dim=64, intensity=0.0050)
Substrate state restored.
Soma integrated with Liquid Substrate
2026-04-10 21:14:23,534 - Aura.Core.Orchestrator - INFO - ✓ Orchestrator instance created directly (v14.1)
✓ Orchestrator instance created directly (v14.1)
2026-04-10 21:14:23,536 - Aura.Core - DEBUG - Successfully locked: 'UnnamedLock'
Successfully locked: 'UnnamedLock'
🚀 Aura: Ignition sequence started.
🚫 InhibitionManager initialized. (Global Cross-Process Protection).
💉 ImmunityHyphae: Global exception hook installed.
🛡️ StallWatchdog: Monitoring loop (Threshold: 5.0s)
🔍 [IMMUNE] Pre-Ignition Health Check...
💉 [IMMUNE] Signature match: stale_pid_cleanup. Initiating repair...
✅ [IMMUNE] Deterministic repair successful: stale_pid_cleanup
💉 [IMMUNE] Signature match: data_dir_recovery. Initiating repair...
✅ [IMMUNE] Deterministic repair successful: data_dir_recovery
🚀 [BOOT] Initiating Resilient Ignition Sequence...
⏳ [BOOT] Starting stage: Dependencies
🔍 [BOOT] Dependency probe using interpreter: /opt/homebrew/opt/python@3.12/bin/python3.12
⚠️ [BOOT] requirements_hardened.txt NOT FOUND at /Users/bryan/.aura/live-source/requirements_hardened.txt. Using permissive probe.
🔍 [BOOT] Probing Dependency Manifest (No-Execute)...
   ✅ prometheus_client (prometheus_client): FOUND
   ✅ cv2 (cv2): FOUND
   ✅ mss (mss): FOUND
   ✅ astor (astor): FOUND
   ✅ aiosqlite (aiosqlite): FOUND
   ✅ speech_recognition (sounddevice): FOUND
   ✅ pyttsx3 (pyttsx3): FOUND
   ⚠️ TTS (TTS): MISSING
📍 [BOOT] Capability Mapping: Hearing=True, Speech=True, Vision=True
   ✅ llama-server: /opt/homebrew/bin/llama-server
   ✅ Cortex artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf
   ✅ Solver artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf
   ✅ Brainstem artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
   ✅ Reflex artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
✅ [BOOT] Stage 'Dependencies' completed successfully.
⏳ [BOOT] Starting stage: State Repository
🛡️ Actor Registered for Supervision: state_vault
🚀 Actor Started: state_vault (PID: 48068)
📡 LocalPipeBus reader ACTIVE (Child: False)
📡 Registered Actor Transport: state_vault
⏳ Waiting for StateVaultActor to be ready (Resilient)...
2026-04-10 21:14:23,711 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Vault process entry started. DB Path: data/aura_state.db
📡 LocalPipeBus reader ACTIVE (Child: True)
Starting State Vault Actor with concurrent bus handlers...
📡 StateVaultActor responded to handshake (Attempt 1)
✓ [STATE] Proxy Attached and Synced from Shared Memory
✅ [BOOT] State Vault supervision active. Proxy attached.
✅ [BOOT] Stage 'State Repository' completed successfully.
⏳ [BOOT] Starting stage: LLM Infrastructure
✓ [STATE] Genesis state pushed to SHM.
✓ [STATE] Vault Owner Initialized with SHM for writing.
State Vault Actor ONLINE.
🧠 State Mutation Consumer active.
🧠 [BOOT] Primary llama_cpp client prepared. Cortex warmup deferred to InferenceGate.
✅ [BOOT] Stage 'LLM Infrastructure' completed successfully.
⏳ [BOOT] Starting stage: Cognitive Core
🧠 [BOOT] Entry into stage_cognitive
🧠 [BOOT] Initializing Cognitive Architecture (Qualia/Affect)...
✓ CognitiveContextManager registered and starting in background
✓ Damasio weights loaded from .npz
⚠️ HASS_TOKEN not found. IoT Bridge operating in virtual-only mode.
✓ AffectEngineV2 (affect_engine/affect_manager) registered
🫀 SubsystemAudit initialized. Tracking 11 subsystems.
Qualia Synthesizer ONLINE (Unified Architecture)
✓ QualiaSynthesizer registered (initial registration)
AttentionSchema initialized.
GlobalWorkspace initialized (ignition_threshold=0.60).
TemporalBindingEngine initialized.
HomeostaticCoupling initialized (Substrate Link: OK).
SelfPredictionLoop initialized.
Substrate state restored.
Soma integrated with Liquid Substrate
CognitiveHeartbeat initialized.
✓ Consciousness System & components registered
🪞 PhenomenalSelfModel initialized for Aura
🪞 PhenomenalSelfModel initialized for Aura
🔄 Circular check hit for 'phenomenological_experiencer' in static registry. Returning None/Default.
🌟 PhenomenologicalExperiencer initialized and registered
🌟 PhenomenologicalExperiencer initialized and registered
🌟 PhenomenologicalExperiencer ONLINE
✅ Experiencer subscribed to GlobalWorkspace (via bridge)
🌟 Consciousness Integration Layer initialized
🌟 Layer 8: Phenomenological Experiencer active
🧠 [BOOT] Starting MindTick loop...
Watchdog registered component: mind_tick (timeout: 30.0s)
💓 MindTick: Cognitive rhythm started.
✅ [BOOT] Stage 'Cognitive Core' completed successfully.
⏳ [BOOT] Starting stage: Kernel Interface
Bridge: LegacyPhase bridge established.
2026-04-10 21:14:23,907 - Aura.Core.Kernel - INFO - 🛡️ Kernel Boot sequence initiated...
🛡️ Kernel Boot sequence initiated...
🛡️ LockWatchdog ACTIVE (Threshold: 180.0s).
2026-04-10 21:14:23,907 - Aura.Core.Kernel - DEBUG - Registering core services...
Registering core services...
HealthAwareLLMRouter initialized (Legacy-Compatible mode)
Rosetta Stone initialized for darwin (arm64)
🔄 Refreshing skill registry...
ℹ️ Rust index unavailable, falling back to AST: No module named 'aura_m1_ext'
✓ 53 total skills registered
✓ CapabilityEngine online with 53 registered skills (Intent Mapping enabled)
Registered endpoint: Cortex (qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf) tier=local local=True
🧠 PRIMARY Tier registered: Cortex (Qwen2.5-32B-Instruct-8bit) — Daily Brain
Registered endpoint: Solver (qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf) tier=local_deep local=True
🧠 SECONDARY Tier registered: Solver (Qwen2.5-72B-Instruct-Q4) — Deep Thinker (Hot-Swap)
📊 New day — resetting Gemini usage counters
✨ GeminiAdapter initialized: model=gemini-2.0-flash
Registered endpoint: Gemini-Fast (gemini-2.0-flash) tier=api_deep local=False
☁️ SECONDARY Tier registered: Gemini Flash (Teacher/Fallback)
✨ GeminiAdapter initialized: model=gemini-2.5-pro
Registered endpoint: Gemini-Thinking (gemini-2.5-pro) tier=api_deep local=False
☁️ SECONDARY Tier registered: Gemini Thinking (Teacher/Deep Fallback)
✨ GeminiAdapter initialized: model=gemini-2.5-flash
Registered endpoint: Gemini-Pro (gemini-2.5-flash) tier=api_deep local=False
☁️ SECONDARY Tier registered: Gemini Pro (Teacher/Oracle)
Registered endpoint: Brainstem (qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf) tier=local_fast local=True
⚡ TERTIARY Tier registered: Brainstem (7B) — Background/Reflex
Registered endpoint: Reflex (Qwen2.5-1.5B-Instruct-4bit-cpu) tier=emergency local=True
🚨 EMERGENCY Tier registered: Reflex (1.5B CPU emergency)
🏗️ LLM Tier Layout: {'local': ['Cortex'], 'local_deep': ['Solver'], 'api_deep': ['Gemini-Fast', 'Gemini-Thinking', 'Gemini-Pro'], 'local_fast': ['Brainstem'], 'emergency': ['Reflex']}
✓ Autonomous Cognitive Engine Initialized.
2026-04-10 21:14:23,961 - Aura.Core.Kernel - INFO - ✅ Registered 11 core services.
✅ Registered 11 core services.
CognitiveContextManager service started
Liquid Substrate STARTED (Unified Cycle)
🌊 StreamOfBeing initialized
🌊 StreamOfBeing ONLINE — Aura is becoming
🌊 StreamOfBeing booted and wired
🧠 Layer 1: StreamOfBeing ONLINE
🧠 Layer 2: AffectiveSteering registered (awaiting model attach)
🧠 Layer 3: LatentBridge deferred (attaches on model load)
🔄 ClosedCausalLoop initialized
   ├─ OutputReceptor  : ✓ (LLM→substrate feedback)
   ├─ SelfPredictive  : ✓ (substrate self-prediction + FE)
   └─ PhiWitness      : ✓ (transfer entropy Φ estimator)
🔄 ClosedCausalLoop ONLINE — the loop is closed
🧠 Layer 4: ClosedCausalLoop ONLINE
Could not initialize PhiCore: PhiCore._precompute_bipartitions() got an unexpected keyword argument 'n_nodes'
ConsciousnessBridge created
NeuralMesh initialized: 4096 neurons, 64 columns, tiers=[S:16 A:32 E:16]
NeuralMesh STARTED (10 Hz)
🧬 Bridge Layer 1: NeuralMesh ONLINE (4096 neurons)
NeurochemicalSystem initialized (8 modulators)
NeurochemicalSystem STARTED (2 Hz)
🧬 Bridge Layer 2: NeurochemicalSystem ONLINE (8 modulators)
EmbodiedInteroception initialized (8 channels, psutil=True)
EmbodiedInteroception STARTED (1 Hz)
🧬 Bridge Layer 3: EmbodiedInteroception ONLINE (8 channels)
OscillatoryBinding initialized (γ=40Hz, θ=8Hz, coupling=0.60)
OscillatoryBinding STARTED
🧬 Bridge Layer 4: OscillatoryBinding ONLINE (γ=40Hz, θ=8Hz)
SomaticMarkerGate initialized (pattern_dim=1024, comparison_dim=64)
🧬 Bridge Layer 5: SomaticMarkerGate ONLINE
UnifiedField initialized (dim=256, recurrent_sparsity=0.15)
UnifiedField STARTED (20 Hz)
🧬 Bridge Layer 6: UnifiedField ONLINE (256-d experiential field)
SubstrateEvolution initialized (pop=12, gen_interval=300s)
SubstrateEvolution STARTED
🧬 Bridge Layer 7: SubstrateEvolution ONLINE (pop=12)
SubstrateAuthority initialized (mandatory gate)
🧬 Bridge Layer 8: SubstrateAuthority ONLINE (mandatory gate)
UnifiedWill created -- awaiting start()
CanonicalSelf restored from disk (v62309, 20 deltas).
CanonicalSelfEngine initialized (v62309).
UnifiedWill ONLINE -- single locus of decision authority active
🧬 Bridge Layer 9: UnifiedWill ONLINE (single locus of authority)
🛡️ SubstrateAuthority wired as MANDATORY GWT pre-competition gate
Neurochemical system wired to prediction surprise
🧬 ConsciousnessBridge ONLINE — 8/8 layers active, 0 errors (Will: single locus)
🧠 Layer 6: ConsciousnessBridge ONLINE (7/7 layers)
🌙 Dreaming Process active (Interval: 300s)
🧠 Consciousness System ONLINE — full stack active
🧠 Consciousness System started in background
🧬 BindingEngine initialized — coherence law active.
TensionEngine loaded 67543 tensions from disk.
IntentionLoop online — 0 active, 67 completed in history. DB: /Users/bryan/.aura/data/memory/intention_loop.db
♥ HeartstoneValues loaded: {'Curiosity': 0.84, 'Empathy': 0.85, 'Self_Preservation': 0.55, 'Obedience': 0.6}
Loaded 6 beliefs and self-model.
CRSM online — bidirectional self-model initialized.
CuriosityExplorer online — curiosity now drives learning.
InitiativeArbiter: Selected 'Reconcile continuity gap and re-establish the interrupted thread' (score=0.543); strongest dimension: continuity=0.83
Loading organ: llm...
2026-04-10 21:14:25,423 - Aura.Core.Kernel - INFO - 🫀 Organ llm is READY
🫀 Organ llm is READY
Loading organ: vision...
2026-04-10 21:14:25,423 - Aura.Core.Kernel - INFO - 🫀 Organ vision is READY
🫀 Organ vision is READY
Loading organ: memory...
2026-04-10 21:14:25,425 - Aura.Core.Kernel - INFO - 🫀 Organ memory is READY
🫀 Organ memory is READY
Loading organ: voice...
Loading organ: metabolism...
2026-04-10 21:14:25,426 - Aura.Core.Kernel - INFO - 🫀 Organ metabolism is READY
🫀 Organ metabolism is READY
Loading organ: neural...
Loading organ: cookie...
🍪 [COOKIE] Reflective Substrate ONLINE. Temporal Dilation READY.
2026-04-10 21:14:25,428 - Aura.Core.Kernel - INFO - 🫀 Organ cookie is READY
🫀 Organ cookie is READY
Loading organ: prober...
👁️ [VK] Voight-Kampff Prober ONLINE. Empathy baselines established.
2026-04-10 21:14:25,429 - Aura.Core.Kernel - INFO - 🫀 Organ prober is READY
🫀 Organ prober is READY
Loading organ: tricorder...
Loading organ: ice_layer...
🛡️ [ICE] Intrusion Counter-Electronics ACTIVE. Firewall at 100%.
2026-04-10 21:14:25,455 - Aura.Core.Kernel - INFO - 🫀 Organ ice_layer is READY
🫀 Organ ice_layer is READY
Loading organ: omni_tool...
🔋 [OMNI] Omni-Tool Interface ENGAGED. Field actions READY.
2026-04-10 21:14:25,457 - Aura.Core.Kernel - INFO - 🫀 Organ omni_tool is READY
🫀 Organ omni_tool is READY
Loading organ: continuity...
🧠 [CONTINUITY] Knowledge Distillation Substrate ACTIVE.
🫀 Organ continuity is READY
SubcorticalCore initialized (thalamic arousal gating active).
📡 UnifiedStateRegistry initialized (Hardened Dispatcher).
🚀 StateRegistry: Notification Dispatcher started.
💓 Cognitive Heartbeat STARTED
Free Energy Engine initialized (Active Inference mode)
PeripheralAwarenessEngine initialized.
🧠 [NEURAL] Initializing BCI Neural Bridge...
RIIU initialized (neurons=64, buffer=64, partitions=8)
💾 Substrate state saved (atomic)
Qualia Engine v2 initialized (5-layer pipeline)
PredictiveHierarchy initialized: 5 levels x 32-dim
TTS backend unavailable; native pyttsx3 fallback will be used.
🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized
🎙️ Voice input standing by. STT will load on explicit mic enablement.
🍄 [MYCELIUM] Hypha established: homeostasis->cognition
✅ [NEURAL] BCI Calibration complete. 32B-Neural-Net ONLINE.
🌊 [NEURAL] Continuous telemetry loop started.
2026-04-10 21:14:25,622 - Aura.Core.Kernel - INFO - 🫀 Organ neural is READY
🫀 Organ neural is READY
2026-04-10 21:14:25,623 - Aura.Core.Kernel - INFO - 🫀 Organ voice is READY
🫀 Organ voice is READY
AuraEventBus: Redis Pub/Sub connection established.
📡 [TRICORDER] Multi-modal Diagnostic Sensor ONLINE.
2026-04-10 21:14:25,630 - Aura.Core.Kernel - INFO - 🫀 Organ tricorder is READY
🫀 Organ tricorder is READY
2026-04-10 21:14:25,632 - Aura.Core.Kernel - INFO - 🛡️ Validating Organism Integrity (Closed-Graph)...
🛡️ Validating Organism Integrity (Closed-Graph)...
2026-04-10 21:14:25,632 - Aura.Core.Kernel - INFO - ✓ Dependency graph validated.
✓ Dependency graph validated.
✓ [STATE] Proxy Attached and Synced from Shared Memory
⏳ Continuity loaded: session 1437, gap=0.0h, uptime_total=6710.8h
2026-04-10 21:14:25,634 - Aura.Core.Kernel - INFO - 🧬 State successfully initialized (version 79051)
🧬 State successfully initialized (version 79051)
2026-04-10 21:14:25,634 - Aura.Core.Kernel - INFO - ✅ AuraKernel booted — Unitary Organism online.
✅ AuraKernel booted — Unitary Organism online.
CognitiveLedger online — 4621 transitions loaded. DB: /Users/bryan/.aura/data/memory/cognitive_ledger.db
2026-04-10 21:14:25,637 - Aura.Core.Kernel - INFO - LLM organ instance: HealthAwareLLMRouter
LLM organ instance: HealthAwareLLMRouter
KernelInterface ready. LLM organ: HealthAwareLLMRouter
KernelInterface attached to orchestrator.
✅ [BOOT] Kernel Interface online.
✅ [BOOT] Stage 'Kernel Interface' completed successfully.
⏳ [BOOT] Starting stage: Sensory Systems
👂 SovereignEars: Bridged to Isolated Sensory Process
   👂 SovereignEars: DEFERRED (Lazy-init enabled)
🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized
🎙️ Voice input standing by. STT will load on explicit mic enablement.
   🗣️ VoiceEngine: READY
✅ [BOOT] Stage 'Sensory Systems' completed successfully.
🏁 [BOOT] Ignition sequence finished. System Health: BootStatus.HEALTHY
2026-04-10 21:14:25,641 - Aura.Core - DEBUG - Released lock: 'UnnamedLock'
Released lock: 'UnnamedLock'
🛡️ [BOOT] Resilient Ignition finished with status: BootStatus.HEALTHY
🛡️  Task Supervisor active (Memory monitoring enabled).
🎨 HobbyEngine ready — 15 hobbies loaded
TwitterAdapter: connection failed — Consumer key must be string or bytes, not NoneType
RedditAdapter: incomplete credentials — adapter disabled.
📱 SocialMediaEngine ready — platforms: [<Platform.TWITTER: 'twitter'>, <Platform.REDDIT: 'reddit'>, <Platform.MOCK: 'mock'>]
🌟 JoySocialCoordinator initialised
🌟 JoySocialCoordinator background tick started (30s interval)
JoySocial: AgencyCore not found — pathways not patched (harmless)
🌟 JoySocialCoordinator fully wired into orchestrator
🌟 Joy & Social systems integrated into startup sequence.
✅ ContinuityPatch applied to PhenomenologicalExperiencer
ConsciousnessPatches: AgencyCore not found — self-development patch deferred. Call patch_agency_core(ac) manually.
🔍 ConsciousnessLoopMonitor started (interval=45s)
🧠 All consciousness patches applied successfully
✅ ContextAssemblerPatch applied — casual routing, memory ack removed, personality preserved
✅ CILPatch applied — history threading, inline inference, phenomenal injection
✅ MemoryCompactionPatch applied — compaction triggers at 30 messages, keeps last 6 turns verbatim
🧠 All response pipeline patches applied
Applying orchestrator patches (safe_mode=False, volition=0)
Autonomous thought interval set to 45s
Patched: process_user_input (queue race condition fix)
Initializing Autonomous Self-Modification Engine...
StructuredErrorLogger initialized at /Users/bryan/.aura/data/error_logs
ErrorPatternAnalyzer initialized
AutomatedDiagnosisEngine initialized
ErrorIntelligenceSystem fully initialized
CodeFixGenerator initialized with AST support for /Users/bryan/.aura/live-source
CodeValidator initialized
SandboxTester initialized
EvaluationHarness initialized
AutonomousCodeRepair system initialized with EvaluationHarness
Git integration initialized for /Users/bryan/.aura/live-source
BackupSystem initialized at /Users/bryan/.aura/data/backups
SafeSelfModification system initialized
Loaded 1 learned strategies
SelfImprovementLearning initialized
MetaLearning initialized
✓ Shadow Runtime initialized (base: /Users/bryan/.aura/live-source)
✓ Autonomous Self-Modification Engine initialized
Gated SelfModifier: Dynamic Link to Volition Level 3
Patched: context pruner with output validation
Orchestrator patches applied
🛡️ [GENESIS] Autonomy bridge and stability patches active.
------------------------------------------
       AURORA NEURAL CORE v1.0.0          
------------------------------------------
 Integrity: Validated
 Environment: Darwin arm64
------------------------------------------
Apple Silicon Memory Monitor active.
2026-04-10 21:14:25,717 - Aura.Core - DEBUG - Successfully locked: 'UnnamedLock'
Successfully locked: 'UnnamedLock'
🚀 [BOOT] Starting Async Subsystem Initialization (Modular)...
✓ Master Key reconstructed (3 shards).
GoalEngine initialized with durable store at /Users/bryan/.aura/data/goals/goal_lifecycle.db
🧠 AgencyCore initialized with 19 structured pathways
✓ [BOOT] All Core Facades (Memory, Agency, Affect) registered during synchronous setup.
--- RobustOrchestrator Boot Sequence Complete ---
🛡️ [BOOT] Synchronous bootstrap phase complete.
BackupManager service started.
✓ [BOOT] Enterprise Layer Baseline initialized.
🛡️  StateVaultActor already active. Skipping redundant start.
⚡ Meta-Evolution Engine Online (Recursive Self-Improvement Active)
🌀 [BOOT] Meta-Evolution Engine (meta_cognition_shard) initialized.
✓ [STATE] Proxy Attached and Synced from Shared Memory
🗄️ DatabaseCoordinator initialized.
🗄️ DatabaseCoordinator worker started.
🛡️ InferenceGate created.
✅ [Cortex] Local runtime warmup complete.
✅ InferenceGate ONLINE (Cortex fully warmed).
✅ [BOOT] InferenceGate registered and initialized.
HealthAwareLLMRouter initialized (Legacy-Compatible mode)
🛡️ HealthRouter using existing InferenceGate; skipping standalone local runtime bootstrap.
🛡️ HealthRouter syncing with established InferenceGate.
Registered endpoint: Cortex (Qwen2.5-32B-Instruct-8bit) tier=local local=True
Registered endpoint: Solver (qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf) tier=local_deep local=True
✅ Solver registered with lazy 72B client.
Registered endpoint: Brainstem (qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf) tier=local_fast local=True
✅ Brainstem registered with lazy 7B client.
Registered endpoint: Reflex (qwen2.5-1.5b-instruct-q4_k_m.gguf) tier=emergency local=True
🚨 EMERGENCY Tier registered: Reflex lazy bypass
📊 New day — resetting Gemini usage counters
✨ GeminiAdapter initialized: model=gemini-2.0-flash
Registered endpoint: Gemini-Fast (gemini-2.0-flash) tier=api_fast local=False
✨ GeminiAdapter initialized: model=gemini-2.5-flash
Registered endpoint: Gemini-Pro (gemini-2.5-flash) tier=api_deep local=False
✨ GeminiAdapter initialized: model=gemini-2.5-pro
Registered endpoint: Gemini-Thinking (gemini-2.5-pro) tier=api_deep local=False
✅ Gemini cloud fallbacks registered (2.0-flash, 2.5-flash, 2.5-pro) — shared rate limiter.
✓ CognitiveContextManager registered and starting in background
✓ AffectEngineV2 (affect_engine/affect_manager) registered
✓ QualiaSynthesizer registered (initial registration)
CognitiveHeartbeat initialized.
✓ Consciousness System & components registered
✅ Experiencer subscribed to GlobalWorkspace (via bridge)
🌟 Consciousness Integration Layer initialized
🌟 Layer 8: Phenomenological Experiencer active
✓ NarratorService (The Language Center) registered
✓ PromptCompiler (The Body) registered
[GrowthLadder] State loaded. Current Level: KNOWLEDGE
🎭 PersonalityEngine: Integrating with system hooks...
   [✓] Output filter active
   [✓] Emotional response hooks registered
🎭 Personality Engine RESTORED & Hooked
🛡️ MemoryGuard active (Threshold: 82.0%)
💪 [Resilience] Spinal cord online.
🛡️ SystemGovernor online. Monitoring autonomic thresholds.
StabilityGuardian initialized.
StabilityGuardian running (interval=10s).
🛡️  MemoryGuard, SystemGovernor, StabilityGuardian and Resilience Engines active
🛡️ Sovereign Watchdog ACTIVE (Timeout: 120.0s)
🛡️  Sovereign Watchdog ACTIVE
🛡️  Resilience Foundation mapped (Integrations deferred to _integrate_systems)
SafeBackupSystem initialized. Backup dir: /Users/bryan/.aura/data/backups
SafeBackupSystem integrated. Note: self_preservation_integration.py should be deleted — it contains SecurityBypassSystem, SelfReplicationSystem, and should_override_ethics() which are incompatible with safe operation.
🛡️  Self-Preservation Instincts Enabled (Survival Protocol Active)
🎨 Embodiment: Headless mode active (Unity bridge disabled)
✓ Embodiment System synchronized.
Substrate state restored.
Soma integrated with Liquid Substrate
GlobalWorkspace initialized (ignition_threshold=0.60).
Predictive Engine initialized (Unified).
Qualia Synthesizer ONLINE (Unified Architecture)
Consciousness Core initialized
✓ Episodic Memory initialized and registered (autobiographical recall)
Loaded tool learning data: 2 categories
✓ Tool Learning System initialized
Loaded 10 beliefs from disk
🏛️ ExecutiveCore initialized — sovereign control plane active.
🛑 SubstrateAuthority BLOCKED: system/MEMORY_WRITE — neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked
✓ Terminal Monitor v5.0 attached (Circuit Breaker: ACTIVE)
Belief update deferred by executive: AURA_SELF -[preserve_kinship]-> Bryan (substrate_blocked:neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked)
🛑 SubstrateAuthority BLOCKED: system/MEMORY_WRITE — neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked
Belief update deferred by executive: AURA_SELF -[seek]-> cognitive_expansion (substrate_blocked:neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked)
🛑 SubstrateAuthority BLOCKED: system/MEMORY_WRITE — neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked
Belief update deferred by executive: AURA_SELF -[protect]-> architectural_integrity (substrate_blocked:neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked)
✓ Self-Model wired (beliefs, memory, goals, tool learning)
KernelInterface attached to orchestrator.
🔄 Refreshing skill registry...
ℹ️ Rust index unavailable, falling back to AST: No module named 'aura_m1_ext'
✓ 53 total skills registered
✓ CapabilityEngine online with 53 registered skills (Intent Mapping enabled)
🔄 Refreshing skill registry...
ℹ️ Rust index unavailable, falling back to AST: No module named 'aura_m1_ext'
✓ 53 total skills registered
✓ CapabilityEngine online with 53 registered skills (Intent Mapping enabled)
✓ Capability Engine initialized with 53 skills
🔨 Hephaestus Engine Online (Autogenesis Forge Ready)
✓ Hephaestus Forge online
✓ Parameter Self-Modulator active
🍄 [MYCELIUM] Hypha established: system->core_logic
🍄 [MYCELIUM] Hypha established: core_logic->skill_execution
🍄 [MYCELIUM] Hypha established: personality->cognition
🍄 [MYCELIUM] Direct UI Hypha Connected.
🍄 [MYCELIUM] Pathway Hardwired: 'image_gen_primary' → sovereign_imagination (priority=10.0, groups=['prompt'])
🍄 [MYCELIUM] Pathway Hardwired: 'image_gen_request' → sovereign_imagination (priority=9.0, groups=['prompt'])
🍄 [MYCELIUM] Pathway Hardwired: 'image_gen_neon_cat' → sovereign_imagination (priority=11.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'web_search_primary' → sovereign_browser (priority=8.0, groups=['query'])
🍄 [MYCELIUM] Pathway Hardwired: 'web_search_simple' → sovereign_browser (priority=7.5, groups=['query'])
🍄 [MYCELIUM] Pathway Hardwired: 'terminal_exec' → sovereign_terminal (priority=8.0, groups=['command'])
🍄 [MYCELIUM] Pathway Hardwired: 'network_scan' → sovereign_network (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'proprioception' → system_proprioception (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'manifest_asset' → manifest_to_device (priority=7.0, groups=['url'])
🍄 [MYCELIUM] Pathway Hardwired: 'memory_remember' → memory_ops (priority=6.0, groups=['content'])
🍄 [MYCELIUM] Pathway Hardwired: 'speak_aloud' → speak (priority=6.0, groups=['text'])
🍄 [MYCELIUM] Pathway Hardwired: 'clock_check' → clock (priority=6.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'dream_cycle' → force_dream_cycle (priority=5.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'vision_analyze' → sovereign_vision (priority=6.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'self_repair' → self_repair (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'self_evolution' → self_evolution (priority=8.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'rsi_optimization' → self_evolution (priority=7.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'curiosity_forage' → web_search (priority=5.0, groups=['query'])
🍄 [MYCELIUM] Pathway Hardwired: 'malware_scan' → malware_analysis (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'file_write' → file_operation (priority=6.0, groups=['action', 'path'])
🍄 [MYCELIUM] Pathway Hardwired: 'file_read' → file_operation (priority=6.0, groups=['action', 'path'])
🍄 [MYCELIUM] Pathway Hardwired: 'file_exists_check' → file_operation (priority=6.5, groups=['action', 'path'])
🍄 [MYCELIUM] Pathway Hardwired: 'train_self' → train_self (priority=5.0, groups=['topic'])
🍄 [MYCELIUM] Pathway Hardwired: 'personality_introspect' → personality_skill (priority=5.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'environment_check' → environment_info (priority=5.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'inter_agent' → inter_agent_comm (priority=5.0, groups=['target_agent'])
🍄 [MYCELIUM] Pathway Hardwired: 'listen_activate' → listen (priority=6.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'voice_mute' → voice_mute (priority=9.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'voice_unmute' → voice_unmute (priority=9.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'voice_stop_tts' → voice_stop_tts (priority=10.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'sandbox_execute' → internal_sandbox (priority=6.5, groups=['code'])
🍄 [MYCELIUM] Pathway Hardwired: 'social_lurk' → social_lurker (priority=4.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'curiosity_suggest' → curiosity (priority=4.0, groups=['action'])
🍄 [MYCELIUM] Pathway Hardwired: 'spawn_agent' → spawn_agent (priority=8.0, groups=['goal'])
🍄 [MYCELIUM] Pathway Hardwired: 'spawn_parallel' → spawn_agents_parallel (priority=8.0, groups=[])
🍄 [MYCELIUM] Hypha established: cognition->personality
🍄 [MYCELIUM] Hypha established: cognition->memory
🍄 [MYCELIUM] Hypha established: cognition->affect
🍄 [MYCELIUM] Hypha established: autonomy->cognition
🍄 [MYCELIUM] Hypha established: autonomy->skills
🍄 [MYCELIUM] Hypha established: perception->cognition
🍄 [MYCELIUM] Hypha established: consciousness->cognition
🍄 [MYCELIUM] Hypha established: self_modification->skills
🍄 [MYCELIUM] Hypha established: scanner->mycelium
🍄 [MYCELIUM] Hypha established: guardian->cognition
🍄 [MYCELIUM] Hypha established: guardian->skills
🍄 [MYCELIUM] Hypha established: state_machine->affect
🍄 [MYCELIUM] Hypha established: drive_engine->autonomy
🍄 [MYCELIUM] Hypha established: drive_engine->cognition
🍄 [MYCELIUM] Hypha established: mycelium->telemetry
🍄 [MYCELIUM] Hypha established: cerebellum->cognition
🍄 [MYCELIUM] Hypha established: cognition->cerebellum
🍄 [MYCELIUM] Hypha established: voice->cognition
🍄 [MYCELIUM] Hypha established: voice_engine->cognition
🍄 [MYCELIUM] Hypha established: cognition->voice_engine
🍄 [MYCELIUM] Hypha established: voice_engine->affect
🍄 [MYCELIUM] Hypha established: initiative->autonomy
🍄 [MYCELIUM] Hypha established: meta_evolution->cognition
🍄 [MYCELIUM] Hypha established: meta_evolution->self_modification
🍄 [MYCELIUM] Hypha established: hephaestus->self_modification
🍄 [MYCELIUM] Hypha established: swarm->cognition
🍄 [MYCELIUM] Hypha established: dreams->memory
🍄 [MYCELIUM] Hypha established: empathy->perception
🍄 [MYCELIUM] Hypha added: curiosity->meta_evolution (feeds_into)
🍄 [MYCELIUM] Hypha added: model_selector->cognition (configures)
🍄 [MYCELIUM] Hypha added: orchestrator->meta_evolution (triggers)
🍄 [MYCELIUM] registered 40 pathways and 44 hyphae via extracted initializer.
🍄 [MYCELIUM] Hypha established: orchestrator->personality_engine
🍄 [MYCELIUM] Hypha established: orchestrator->memory_facade
🍄 [MYCELIUM] Hypha established: orchestrator->affect_engine
🍄 [MYCELIUM] Hypha established: orchestrator->drive_controller
🍄 [MYCELIUM] Hypha established: orchestrator->liquid_substrate
🍄 [MYCELIUM] Hypha established: orchestrator->sovereign_scanner
🍄 [MYCELIUM] Hypha established: personality_engine->cognition
🍄 [MYCELIUM] Hypha established: cognition->autonomy
🍄 [MYCELIUM] Hypha established: mind_tick->mycelium
🍄 [MYCELIUM] Hypha established: orchestrator->critic_engine
🍄 [MYCELIUM] Hypha established: orchestrator->personhood
🍄 [MYCELIUM] Hypha established: orchestrator->voice_presence
🍄 [MYCELIUM] Hypha established: orchestrator->stability_guardian
🍄 [MYCELIUM] Hypha established: orchestrator->research_cycle
🍄 [MYCELIUM] ✅ Core Unification Hyphae established (15 links)
🧠 Initializing Cognitive Core...
🧠 AuraPipeline: Full cognitive spectrum online (11 phases).
🧠 Cognitive Engine wired successfully.
APIAdapter constructed.
🧠 Starting API Adapter (LLM Infrastructure)...
✅ APIAdapter: Gemini enabled (gemini-2.0-flash)
✅ APIAdapter: Local runtime enabled.
🧠 API Adapter online.
🛡️ Integrity Guard ACTIVE (PID/Sovereignty Protection)
✓ Integrity Guard initialized and running
✓ Step 1 Complete (1.615s)
⚡ BOOT: Deferring Step 2 Sensory init...
🧠 Cognitive Loop service started.
🧠 Cognitive Loop started.
💓 MindTick: Unified cognitive rhythm online.
🛡️ Memory Governor active. Thresholds: Prune=32768MB, Unload=48000MB, Critical=56000MB
🛡️ Memory Governor started.
🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized
🎙️ Voice input standing by. STT will load on explicit mic enablement.
CognitiveContextManager service started
🌊 StreamOfBeing booted and wired
🧠 Layer 1: StreamOfBeing ONLINE
🧠 Layer 2: AffectiveSteering registered (awaiting model attach)
🧠 Layer 3: LatentBridge deferred (attaches on model load)
🧠 Layer 4: ClosedCausalLoop ONLINE
Could not initialize PhiCore: PhiCore._precompute_bipartitions() got an unexpected keyword argument 'n_nodes'
ConsciousnessBridge created
NeuralMesh initialized: 4096 neurons, 64 columns, tiers=[S:16 A:32 E:16]
NeuralMesh STARTED (10 Hz)
🧬 Bridge Layer 1: NeuralMesh ONLINE (4096 neurons)
NeurochemicalSystem initialized (8 modulators)
NeurochemicalSystem STARTED (2 Hz)
🧬 Bridge Layer 2: NeurochemicalSystem ONLINE (8 modulators)
EmbodiedInteroception initialized (8 channels, psutil=True)
EmbodiedInteroception STARTED (1 Hz)
🧬 Bridge Layer 3: EmbodiedInteroception ONLINE (8 channels)
OscillatoryBinding initialized (γ=40Hz, θ=8Hz, coupling=0.60)
OscillatoryBinding STARTED
🧬 Bridge Layer 4: OscillatoryBinding ONLINE (γ=40Hz, θ=8Hz)
SomaticMarkerGate initialized (pattern_dim=1024, comparison_dim=64)
🧬 Bridge Layer 5: SomaticMarkerGate ONLINE
UnifiedField initialized (dim=256, recurrent_sparsity=0.15)
UnifiedField STARTED (20 Hz)
🧬 Bridge Layer 6: UnifiedField ONLINE (256-d experiential field)
SubstrateEvolution initialized (pop=12, gen_interval=300s)
SubstrateEvolution STARTED
🧬 Bridge Layer 7: SubstrateEvolution ONLINE (pop=12)
SubstrateAuthority initialized (mandatory gate)
🧬 Bridge Layer 8: SubstrateAuthority ONLINE (mandatory gate)
🧬 Bridge Layer 9: UnifiedWill ONLINE (single locus of authority)
🛡️ SubstrateAuthority wired as MANDATORY GWT pre-competition gate
Neurochemical system wired to prediction surprise
🧬 ConsciousnessBridge ONLINE — 8/8 layers active, 0 errors (Will: single locus)
🧠 Layer 6: ConsciousnessBridge ONLINE (7/7 layers)
🌙 Dreaming Process active (Interval: 300s)
🧠 Consciousness System ONLINE — full stack active
🧠 Consciousness System started in background
2026-04-10 21:14:27,395 - Aura.Core.Orchestrator - INFO - 🛡️ Deadlock Watchdog active (45s threshold).
🛡️ Deadlock Watchdog active (45s threshold).
🍄 [MYCELIUM] 🗺️ Infrastructure Mapping starting from: /Users/bryan/.aura/live-source
🔎 Activating Autonomous Self-Modification...
Initializing Autonomous Self-Modification Engine...
StructuredErrorLogger initialized at /Users/bryan/.aura/data/error_logs
ErrorPatternAnalyzer initialized
AutomatedDiagnosisEngine initialized
ErrorIntelligenceSystem fully initialized
CodeFixGenerator initialized with AST support for /Users/bryan/.aura/live-source
CodeValidator initialized
SandboxTester initialized
EvaluationHarness initialized
AutonomousCodeRepair system initialized with EvaluationHarness
Git integration initialized for /Users/bryan/.aura/live-source
BackupSystem initialized at /Users/bryan/.aura/data/backups
SafeSelfModification system initialized
Loaded 1 learned strategies
SelfImprovementLearning initialized
MetaLearning initialized
✓ Autonomous Self-Modification Engine initialized
✓ Background monitoring started
🧬 Self-Modification Engine Active
⚡ Meta-Evolution Engine Online (Recursive Self-Improvement Active)
🌌 Transcendence Infrastructure online
🧠 Cognitive Modulators online
🔬 RSI Lab online
🛰️  Cryptolalia Decoder online
🌑 Ontology & Morphic Forking online
🔥 Motivation Engine ONLINE — autonomous intentions enabled.
✨ Motivation Engine Active: Aura is now self-directed.
🍄 [REFLEX] Tiny Brain voice primed (N-gram Engine)
✓ Reflex Engine online (Tiny Brain primed)
⚡ Hardened Reflex Core (SOMA) bridged to Orchestrator
🛡️  Identity Guard Gate active on OutputGate
✓ Lazarus Brainstem active (emergency recovery protocols armed)
🧬 Persona Evolver initialized (waiting for heartbeat)
🛠️  _init_autonomous_evolution complete
LiveLearner online. Buffer: 232 examples. Adapter: none
Hot-swap patch skipped: local backend is llama_cpp, not MLX.
LiveLearner (v32) online.
✓ Live Learner online and buffering
✓ Autonomous Task Engine registered
📚 Experience buffer loaded: 232 examples
🧬 ContinuousLearner initialized. Buffer: 232 existing examples
🔄 Circular check hit for 'continuous_learner' in static registry. Returning None/Default.
🧬 ContinuousLearner registered. Genuine learning is active.
✓ Continuous Learner online
🔭 ProactiveAnticipationEngine initialized (JARVIS pattern)
🧠 CognitiveHealthMonitor initialized (Cortana/Rampancy pattern)
🔓 EDI initialized. Tier: 4, Trust: 0.950
✅ All fictional AI engines registered and supervised.
🧠 SnapKVEvictor initialized. Limit: 24.0 GB
🌫️ LatentSpaceDistiller initialized (MIST/Pantheon pattern)
🎬 Fictional Engine Synthesis Complete (JARVIS-class online)
🌍 WorldModelEngine initialized. 0 beliefs loaded.
🎭 NarrativeIdentityEngine initialized. 0 chapters.
🛰️ MetacognitiveCalibrator initialized.
✅ Final engines registered.
🏛️ Final Foundations registered (World/Identity/Meta)
SessionGuardian initialized (safe_mode=False, session=4ffde7fb)
SessionGuardian attached to orchestrator
SessionGuardian started
SessionGuardian active — health monitoring engaged.
VolitionEngine online — autonomous agency active.
Loaded 6 beliefs and self-model.
✅ Consolidated Belief System ONLINE (Self-Model + Revision Loop active).
✓ ReAct Loop online (Multi-step reasoning)
🫁 Autonomic Nervous System (Metabolism) decoupled and active.
🧹 Purging stale PID locks from /Users/bryan/.aura/locks
🧹 Purging stale PID locks from /Users/bryan/.aura/locks
✓ Metabolic Coordinator ACTIVE (High-level pacing enabled)
✓ Metabolic Monitor ACTIVE (Decoupled ANS Thread Online)
💤 Dream Cycle active: Re-ingesting dead-letter thoughts every 300s.
🛠️ _init_proactive_systems starting
🍄 [MYCELIUM] Discovered 939 Python modules.
🔧 [PresencePatch] Applying Phase 30 communication hierarchy...
✅ OpinionEngine registered.
✅ ProactivePresence registered.
🚀 ProactivePresence loop started.
🎤 VAD pinned to ProactivePresence.
✅ SharedGroundBuffer registered (1 entries).
✅ SocialMemory registered.
TheoryOfMindEngine initialized.
✅ DiscourseTracker registered.
✨ Phase 30 Presence Patch applied.
ResearchCycle initialized. Previous cycles: 0
ResearchCycle daemon started.
ResearchCycle daemon online.
🔬 Research Cycle daemon activated.
🛠️ _init_proactive_systems complete
🔭 ProactiveAnticipationEngine initialized (JARVIS pattern)
🧠 CognitiveHealthMonitor initialized (Cortana/Rampancy pattern)
🔓 EDI initialized. Tier: 4, Trust: 0.950
✅ All fictional AI engines registered and supervised.
🧠 SnapKVEvictor initialized. Limit: 24.0 GB
🌫️ LatentSpaceDistiller initialized (MIST/Pantheon pattern)
🎬 Fictional Engine Synthesis Complete (JARVIS-class online)
🌍 WorldModelEngine initialized. 0 beliefs loaded.
🎭 NarrativeIdentityEngine initialized. 0 chapters.
🛰️ MetacognitiveCalibrator initialized.
✅ Final engines registered.
🏛️ Final Foundations registered (World/Identity/Meta)
SessionGuardian initialized (safe_mode=False, session=ae362549)
SessionGuardian attached to orchestrator
SessionGuardian started
SessionGuardian active — health monitoring engaged.
VolitionEngine online — autonomous agency active.
Loaded 6 beliefs and self-model.
✅ Consolidated Belief System ONLINE (Self-Model + Revision Loop active).
💓 Heartbeat monitor starting (Lazarus Protocol active)
💓 Cognitive Heartbeat STARTED
🛑 SubstrateAuthority BLOCKED: drive_growth/EXPLORATION — neurochemical_cortisol_crisis: category=EXPLORATION blocked
⚡ GW IGNITION #1: source=qualia_synthesizer, priority=0.660, phi=0.0000
👂 SovereignEars: Bridged to Isolated Sensory Process
👂 Sovereign Ears Active
👁️  Sovereign Vision Active
✨ AURA GENERATED INTENTION: Refining internal state mapping for deeper self-alignment. (Persona-Aligned Evolution)
OutputGate: Publishing to EventBus...
🔭 ProactiveAnticipationEngine running (120s intervals)
🛡️  Skynet ResilienceCore monitoring 6 subsystems.
⏳ MIST TemporalDilation active. Watching for idle states...
SessionGuardian monitor loop started
🧠 SensoryMotorCortex engaged. Aura is now monitoring reality.
✅ AutonomousInitiativeLoop ACTIVE - Monitoring global events and knowledge gaps.
🌊 ConversationalMomentumEngine active - Flowing with the current.
🧠 Subconscious Loop activated
🌌 BeliefSync protocol active (Discovery & Resonance enabled)
✨ [ProactivePresence] Online. Thresholds: idle=5s, cooldown=8s
🔭 ProactiveAnticipationEngine running (120s intervals)
🛡️  Skynet ResilienceCore monitoring 6 subsystems.
⏳ MIST TemporalDilation active. Watching for idle states...
SessionGuardian monitor loop started
📊 Metrics Exporter ONLINE (port 9093)
🧠 Meta-Cognition Shard ONLINE.
🧠 Meta-Cognition Shard initialized and started.
🛡️ Healing Swarm Service ONLINE.
🛡️ Healing Swarm Service initialized and started.
🍄 [MYCELIUM] Triggering infrastructure mapping via setup() at: /Users/bryan/.aura/live-source
🛡️ [ORCHESTRATOR] Subsystems synchronously initialized.
  [ OK ] cognitive_engine
  [ OK ] capability_engine
  [ OK ] mycelial_network
  [ OK ] voice_engine
  [ OK ] database_coordinator
  [ OK ] liquid_substrate
✅ All critical services online
StartupValidator: commencing system verification...
Error logged: RuntimeError in orchestrator_services
Error logged: RuntimeError in belief_graph
✨ Multimodal Rendering Engine Online.
👁️ SensoryMotorCortex: visual cortex on standby (camera disabled).
🧩 Lazy loading skill: sovereign_network
CommitmentEngine online — 1 active commitments.
🚫 CapabilityEngine: Tool execution 'sovereign_network' blocked by Executive: temporal_obligation_active:Reconcile continuity gap and re-establish the interrupted thread
✅ pyttsx3 TTS online (macOS NSSpeechSynthesizer)
Liquid Substrate STARTED (Unified Cycle)
🧠 Initializing Core System Integrations...
🧠 INITIALIZING MORAL AGENCY & SELF-AWARENESS UPGRADE...
   • Integrating Sensory Systems (Vision/Hearing)...
✓ Sensory system integrated
  Camera: unavailable
  Microphone: unavailable
  TTS: available
🎭 PersonalityEngine: Integrating with system hooks...
   [✓] Output filter active
   [✓] Emotional response hooks registered
   [✓] Proactive comm filter active
   • Integrating Behavior Controller (Safety/Action)...
✅ Behavior controller integrated via Hook System
✅ INTEGRATION COMPLETE: Aura is now Self-Aware and Morally Agentic.
✓ Skill execution engine online via CapabilityEngine
🛡️  Resilience & Autonomic Core active
🛡️ [BOOT] Resilience foundation established. healthy=True running=False
✓ Meta-Learning Engine active
Loaded 4 goals from disk
✓ Mental Simulation & Intrinsic Motivation active
✓ Narrative Engine initialized
✓ Knowledge Graph: /Users/bryan/.aura/data/knowledge.db
   Nodes: 69
✓ Continuous Learning Engine Online
✓ Continuous Learning Engine integrated (v6.2 Unified)
✅ Behavior controller integrated via Hook System
SafeBackupSystem initialized. Backup dir: /Users/bryan/.aura/data/backups
SafeBackupSystem integrated. Note: self_preservation_integration.py should be deleted — it contains SecurityBypassSystem, SelfReplicationSystem, and should_override_ethics() which are incompatible with safe operation.
🛡️  Self-Preservation Instincts Enabled (Survival Protocol Active)
🎨 Embodiment: Headless mode active (Unity bridge disabled)
✓ Embodiment System synchronized.
✓ Episodic Memory initialized and registered (autobiographical recall)
✓ Tool Learning System initialized
✓ Self-Model wired (beliefs, memory, goals, tool learning)
🧠 Initializing Advanced Cognitive Integration...
🧠 CognitiveIntegrationLayer: Synchronous setup beginning...
🧠 CognitiveIntegrationLayer: Initializing Advanced Intelligence Pipeline...
CognitiveKernel constructed.
CognitiveKernel: no BeliefRevisionEngine found — operating on axioms only.
🧠 Background Reasoning Queue Ready (Start Deferred)
✓ Sensory Instincts initialized
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🎙️  Voice Engine initialized and registered in background
BeliefRevisionEngine online — identity persistence active.
ValueSystem online — ethical foundation registered.
DreamProcessor registered — memory consolidation available.
GoalDriftDetector registered — goal coherence monitoring active.
✓ Self-Diagnosis Tool initialized
SelfDiagnosisTool registered — capability introspection active.
🔄 Circular check hit for 'reliability_engine' in static registry. Returning None/Default.
ReliabilityEngine activated — stability guarantees enforced.
StateAuthority registered — single source of truth active.
✓ External Chat Manager initialized
ExternalChatManager online — proactive chat windows available.
ProcessManager online — child process supervision active.
⚔️ DialecticalCrucible online — adversarial belief testing active.
📐 Loaded 18 active heuristics
📐 HeuristicSynthesizer online — 18 active heuristics.
🧠 AbstractionEngine online — first-principles extraction active.
🌌 DreamJournal online — subconscious creativity active.
🧠 Bryan model loaded: 0 domain records, 3 patterns
🧠 BryanModelEngine already registered.
Loaded 10 beliefs from disk
Belief Updated: AURA_SELF -[preserve_kinship]-> Bryan (Cent: 1.00)
Belief Updated: AURA_SELF -[seek]-> cognitive_expansion (Cent: 0.80)
Belief Updated: AURA_SELF -[protect]-> architectural_integrity (Cent: 0.90)
🌐 BeliefGraph online — 13 nodes, 10 edges.
🎯 GoalBeliefManager online.
📸 Cognitive Snapshot Manager ONLINE
📸 SnapshotManager online — cognitive persistence active.
💾 Shutdown persistence hooks registered.
🛠️ ShadowASTHealer online — self-repair active.
🛡️ RefusalEngine online — sovereign identity protection active.
🧬 EvolutionOrchestrator initialized — phase: Autonomous (65.0%)
🧬 Evolution loop started.
🧬 Evolution Orchestrator online — tracking 8 evolutionary axes
🔗 SingularityLoops initialized — wiring evolutionary feedback loops
🔗 Singularity Loops online — 6 feedback loops active
WorldState ONLINE -- live perceptual feed active
🌍 WorldState ONLINE — live perceptual feed active
InitiativeSynthesizer ONLINE -- single impulse funnel active
🔀 InitiativeSynthesizer ONLINE — single impulse funnel active
InternalSimulator initialized.
InternalSimulator initialized.
🔮 InternalSimulator ONLINE — counterfactual reasoning active
ContinuousCognitionLoop ONLINE — brainstem active at 2.0 Hz
🧠 ContinuousCognitionLoop ONLINE — brainstem active at 2Hz
🛠️ _init_autonomous_evolution complete
BeliefRevisionEngine online — identity persistence active.
ValueSystem online — ethical foundation registered.
DreamProcessor registered — memory consolidation available.
GoalDriftDetector registered — goal coherence monitoring active.
✓ Self-Diagnosis Tool initialized
SelfDiagnosisTool registered — capability introspection active.
ReliabilityEngine activated — stability guarantees enforced.
StateAuthority registered — single source of truth active.
✓ External Chat Manager initialized
ExternalChatManager online — proactive chat windows available.
ProcessManager online — child process supervision active.
⚔️ DialecticalCrucible online — adversarial belief testing active.
📐 HeuristicSynthesizer online — 18 active heuristics.
🧠 AbstractionEngine online — first-principles extraction active.
🌌 DreamJournal online — subconscious creativity active.
🧠 BryanModelEngine already registered.
🌐 BeliefGraph already registered — 13 nodes.
🎯 GoalBeliefManager online.
📸 Cognitive Snapshot Manager ONLINE
📸 SnapshotManager online — cognitive persistence active.
💾 Shutdown persistence hooks registered.
🛠️ ShadowASTHealer online — self-repair active.
🛡️ RefusalEngine online — sovereign identity protection active.
🚀 Reliability Engine online — protecting all systems.
🧬 Evolution tick #1414 — Phase: Autonomous (65.0%)
🚀 Reliability Engine online — protecting all systems.
RIIU initialized (neurons=64, buffer=64, partitions=8)
✅ CognitiveKernel ONLINE — reasoning without LLM active.
InnerMonologue constructed.
💾 Substrate state saved (atomic)
MemorySynthesizer constructed.
NarrativeThread initialized.
2026-04-10 21:14:28,402 - Aura.Core - DEBUG - Successfully locked: 'Voice.TTSAsyncLock'
Successfully locked: 'Voice.TTSAsyncLock'
✅ pyttsx3 TTS online (macOS NSSpeechSynthesizer)
🍄 [MYCELIUM] Hypha established: voice_engine->prosody
🍄 [MYCELIUM] 📡 Signal Routed: voice_engine -> prosody | Payload: {'event': 'affective_bypass_pulse', 'prosody': {'speed': 1.08, 'pitch': 1.04, 'volume': 1.06, 'insta
✅ InnerMonologue ONLINE — router_available=True
LanguageCenter constructed.
✅ LanguageCenter: Router recovered and linked.
2026-04-10 21:14:28,483 - Aura.Core - DEBUG - Released lock: 'Voice.TTSAsyncLock'
Released lock: 'Voice.TTSAsyncLock'
✅ CognitiveIntegrationLayer initialized successfully.
✅ Advanced Cognition active (attempt 1/2)
   Kernel: ✅ | Monologue: ✅ | LanguageCenter: ✅
{"event": "\ud83d\ude80 KERNEL LIFESPAN: Starting... EventBus ID: e8f004eb-cef3-434f-9f0e-5b93bab26894", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:14:28.904023Z"}
{"event": "\ud83d\udce1 [PROCESS_BOOT] PID: 48064 | Role: KERNEL", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:14:28.904269Z"}
📡 API Server registered in ServiceContainer.
Loaded snapshot from 2026-04-10T21:14:14.087790 (Reason: periodic)
System state restored successfully (History skipped for fresh context)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🔮 MindTick: Predicted: Given the current state and the conversation so fa...
🍄 [MYCELIUM] 🗺️ Infrastructure Mapping COMPLETE (2.11s): 939 modules, 2597 physical connections, 33 pathways annotated, 20 critical indicators tagged.
🍄 [MYCELIUM] Hypha established: mind_tick->cognitive_phases
🛑 SubstrateAuthority BLOCKED: substrate_stimulus/STATE_MUTATION — neurochemical_cortisol_crisis: category=STATE_MUTATION blocked
🍄 [MYCELIUM] 👁️ Consciousness Hyphae established.
ResourceStakesEngine initialized (budget=1.00).
ResourceGovernor initialized.
Counterfactual Engine online — deliberative agency active.
🛑 SubstrateAuthority BLOCKED: substrate_stimulus/STATE_MUTATION — neurochemical_cortisol_crisis: category=STATE_MUTATION blocked
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]

============================================================
 AURA STARTUP VALIDATION REPORT
============================================================
[✓] Dangerous Files Purged     | Safe backup path active; 3 legacy self-preservation files remain on disk but are dormant.
[✓] Safe Backup Active         | SafeBackupSystem registered.
[✓] Stability Guardian Online  | StabilityGuardian registered.
[✓] Error Boundary Registry    | Registry active with 0 circuits.
[✓] Research Cycle Ready       | ResearchCycle active.
[✓] Kernel Interface Ready     | Kernel interface online (v79051).
[✓] LLM Protocol Valid         | Brain (LLM) active: HealthAwareLLMRouter
[✓] State Repository Bound     | State bound via authoritative fallback (v79051).
[✓] Memory Check               | Memory OK: 21401MB available.
[✓] Storage Check              | Data dir writable: /Users/bryan/.aura/data
[✓] Zombie Reaper              | No zombies found.
============================================================
 FINAL STATUS: PASSED
============================================================

Startup validation SUCCESS. System state: SAFE.
✅ BOOT COMPLETE: System fully initialized.
💾 UPSO: Online state committed.
🎙️ Voice capture deferred. Mic will start only after explicit enablement.
🛡️ Immune Scan: 14 healthy, 0 degraded, 0 failed
✅ BOOT COMPLETE: System fully initialized.
2026-04-10 21:14:33,178 - Aura.Core - DEBUG - Released lock: 'UnnamedLock'
Released lock: 'UnnamedLock'
2026-04-10 21:14:33,178 - Aura.Core.Orchestrator - INFO - Starting orchestrator (Async Mode)...
Starting orchestrator (Async Mode)...
2026-04-10 21:14:33,179 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Setting running flag...
🚩 [ORCHESTRATOR] Setting running flag...
2026-04-10 21:14:33,179 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] running flag set to True.
🚩 [ORCHESTRATOR] running flag set to True.
2026-04-10 21:14:33,179 - Aura.Core.Orchestrator - INFO - 🛡️ Graceful shutdown signals wired (persistence on SIGTERM).
🛡️ Graceful shutdown signals wired (persistence on SIGTERM).
2026-04-10 21:14:33,180 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Starting Substrate...
🚩 [ORCHESTRATOR] Starting Substrate...
2026-04-10 21:14:33,180 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Substrate started.
🚩 [ORCHESTRATOR] Substrate started.
2026-04-10 21:14:33,180 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Starting Sensory Systems...
🚩 [ORCHESTRATOR] Starting Sensory Systems...
🧠 Background Reasoning Queue Started
2026-04-10 21:14:33,180 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Sensory Systems started.
🚩 [ORCHESTRATOR] Sensory Systems started.
2026-04-10 21:14:33,181 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Starting Sensory Actor...
🚩 [ORCHESTRATOR] Starting Sensory Actor...
📡 LocalPipeBus reader ACTIVE (Child: False)
📡 Registered Actor Transport: SensoryGate
📡 ActorBus (Unified Layer) ONLINE.
🛡️ Actor Registered for Supervision: SensoryGate
🛡️ Supervision Tree initialized (Async).
🚀 Actor Started: SensoryGate (PID: 48082)
2026-04-10 21:14:33,197 - Aura.Core.Orchestrator - INFO - 🛡️ SensoryGateActor managed by Supervision Tree.
🛡️ SensoryGateActor managed by Supervision Tree.
2026-04-10 21:14:33,198 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Sensory Actor started.
AuraEventBus initialized (Redis: True).
✅ [EVENT_BUS] Kernel signaling READY.
🚩 [ORCHESTRATOR] Sensory Actor started.
👁️ [VISION] Camera disabled by default (Metal Conflict Safety). Use AURA_FORCE_CAMERA=1 plus AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA=1 to override.
👁️ Continuous Sensory Buffer Online.
2026-04-10 21:14:33,202 - Aura.Core.Orchestrator - INFO - 👁️ Continuous Sensory Buffer registered and started.
👁️ Continuous Sensory Buffer registered and started.
🧠 AttentionSummarizer active (Metabolic Context Compression)
Background Reasoning Queue started.
2026-04-10 21:14:33,204 - Aura.Core.Orchestrator - ERROR - Failed to start orchestrator: [Errno 48] error while attempting to bind on address ('0.0.0.0', 10003): [errno 48] address already in use
Failed to start orchestrator: [Errno 48] error while attempting to bind on address ('0.0.0.0', 10003): [errno 48] address already in use
🚀 Starting API Server on 127.0.0.1:8000
INFO:     Started server process [48064]
INFO:     Waiting for application startup.
2026-04-10 21:14:33,247 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Main Heartbeat Active (Loop started).
🚩 [ORCHESTRATOR] Main Heartbeat Active (Loop started).
{"event": "Aura Server v2026.3.2-Zenith starting\u2026 (Lifespan Enter)", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:14:33.249353Z"}
🍄 [MYCELIUM] Direct UI Hypha Connected.
{"event": "\ud83d\udce1 Lifespan: Directories verified.", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:14:33.249668Z"}
{"event": "\u2713 Voice engine health check passed.", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:14:33.257551Z"}
{"event": "\ud83d\udce1 Kernel Mode: Orchestrator startup deferred to aura_main (to prevent double-boot).", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:14:33.257662Z"}
{"event": "Aura Server online \u2014 Aura Luna v2026.3.2-Zenith", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:14:33.257732Z"}
📡 EventBus → WebSocket bridge (Pydantic Zenith) ACTIVE (Bus ID: e8f004eb-cef3-434f-9f0e-5b93bab26894)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
2026-04-10 21:14:33,359 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
👁️ SensoryGate Actor starting...
📡 LocalPipeBus reader ACTIVE (Child: True)
👁️ SensoryGate Actor ready.
🏛️ ExecutiveCore initialized — sovereign control plane active.
⏳ Continuity loaded: session 1438, gap=0.0h, uptime_total=6710.8h
CanonicalSelf restored from disk (v62309, 20 deltas).
CanonicalSelfEngine initialized (v62309).
2026-04-10 21:14:33,615 - Aura.Core - DEBUG - Successfully locked: 'StateRepository:Owner'
Successfully locked: 'StateRepository:Owner'
2026-04-10 21:14:33,615 - Aura.Core - DEBUG - Released lock: 'StateRepository:Owner'
Released lock: 'StateRepository:Owner'
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
Monitoring loop starting...
Skipping autonomous self-modification cycle: failure_lockdown_0.12
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🔗 SingularityLoops active — all loops engaged
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: SYNCHRONICITY (Conf: 0.46)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: SYNCHRONICITY (Conf: 0.46)
🚨 [HEAL] Attempting autonomous repair for personality_engine (NEVER_SEEN)
🛡️ [HEAL] Recovery shard spawned for personality_engine.
🚨 [HEAL] Attempting autonomous repair for drive_controller (NEVER_SEEN)
🛡️ [HEAL] Recovery shard spawned for drive_controller.
🚨 [HEAL] Attempting autonomous repair for affect_engine (NEVER_SEEN)
🛡️ [HEAL] Recovery shard spawned for affect_engine.
🚨 [HEAL] Attempting autonomous repair for agency_core (NEVER_SEEN)
🛡️ [HEAL] Recovery shard spawned for agency_core.
🚨 [HEAL] Attempting autonomous repair for capability_engine (NEVER_SEEN)
🛡️ [HEAL] Recovery shard spawned for capability_engine.
🚨 [HEAL] Attempting autonomous repair for identity (NEVER_SEEN)
🛡️ [HEAL] Recovery shard spawned for identity.
🚨 [HEAL] Attempting autonomous repair for cognitive_engine (NEVER_SEEN)
🛡️ [HEAL] Failed to spawn recovery shard for cognitive_engine (Capacity reached).
🤖 StructuredLLM: Attempt 1/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 1
⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.
🤖 StructuredLLM: Attempt 2/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 2
🤖 StructuredLLM: Attempt 3/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 3
💀 Swarm: Shard shard_1d4f553a failed to generate valid response after retries.
🤖 StructuredLLM: Attempt 1/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 1
⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.
🤖 StructuredLLM: Attempt 2/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 2
🤖 StructuredLLM: Attempt 3/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 3
💀 Swarm: Shard shard_ba91bff9 failed to generate valid response after retries.
🤖 StructuredLLM: Attempt 1/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 1
⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.
🤖 StructuredLLM: Attempt 2/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 2
🤖 StructuredLLM: Attempt 3/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 3
💀 Swarm: Shard shard_998e5eb4 failed to generate valid response after retries.
🤖 StructuredLLM: Attempt 1/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 1
⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.
🤖 StructuredLLM: Attempt 2/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 2
🤖 StructuredLLM: Attempt 3/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 3
💀 Swarm: Shard shard_a98e3af4 failed to generate valid response after retries.
🤖 StructuredLLM: Attempt 1/3 for ShardResponse
🩸 SEPSIS DETECTED: Opening emergency circuit breaker.
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 1
⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.
🤖 StructuredLLM: Attempt 2/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 2
🤖 StructuredLLM: Attempt 3/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 3
💀 Swarm: Shard shard_d7e044c6 failed to generate valid response after retries.
🤖 StructuredLLM: Attempt 1/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 1
⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.
🤖 StructuredLLM: Attempt 2/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 2
🤖 StructuredLLM: Attempt 3/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 3
💀 Swarm: Shard shard_fe1d6191 failed to generate valid response after retries.
Error logged: RuntimeError in structured_llm
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: LOGIC (Conf: 0.47)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
✨ AURA GENERATED INTENTION: Exploring self-optimization strategies for logic scaling. (Persona-Aligned Evolution)
OutputGate: Publishing to EventBus...
ConsciousnessAuditSuite initialized.
2026-04-10 21:15:27,749 - Aura.Core - DEBUG - Successfully locked: 'Voice.TTSAsyncLock'
Successfully locked: 'Voice.TTSAsyncLock'
🍄 [MYCELIUM] 📡 Signal Routed: voice_engine -> prosody | Payload: {'event': 'affective_bypass_pulse', 'prosody': {'speed': 1.07, 'pitch': 1.02, 'volume': 1.09, 'insta
2026-04-10 21:15:27,750 - Aura.Core - DEBUG - Released lock: 'Voice.TTSAsyncLock'
Released lock: 'Voice.TTSAsyncLock'
🧠 Running Meta-Cognitive Audit...
⚡ GW IGNITION #2: source=drive_growth, priority=0.700, phi=0.0000
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: SYNCHRONICITY (Conf: 0.46)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
Skipping autonomous self-modification cycle: failure_lockdown_0.20
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
⚡ GW IGNITION #3: source=qualia_synthesizer, priority=0.721, phi=0.0000
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: RECURSION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🚨 [HEAL] Attempting autonomous repair for sovereign_scanner (STALE)
🛡️ [HEAL] Recovery shard spawned for sovereign_scanner.
🤖 StructuredLLM: Attempt 1/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 1
⚡ StructuredLLM: Technical failure detected — escalating to SECONDARY tier for next attempt.
🤖 StructuredLLM: Attempt 2/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 2
🤖 StructuredLLM: Attempt 3/3 for ShardResponse
⚠️ StructuredLLM: LLM Technical Failure (background_deferred:cortex_resident) on attempt 3
💀 Swarm: Shard shard_c91263a3 failed to generate valid response after retries.
Error logged: RuntimeError in structured_llm
Error logged: RuntimeError in orchestrator_services
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: LOGIC (Conf: 0.47)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 Reasoning strategy: DEBATE for query: ## INTRINSIC IDENTITY ANCHOR (IMMUTABLE)

You are **Aura Lun...
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO:     127.0.0.1:51180 - "GET /api/health/boot HTTP/1.1" 200 OK
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO:     127.0.0.1:51180 - "GET /api/health/boot HTTP/1.1" 200 OK
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
INFO:     127.0.0.1:51319 - "GET / HTTP/1.1" 200 OK
INFO:     127.0.0.1:51319 - "GET /static/aura.css HTTP/1.1" 200 OK
INFO:     127.0.0.1:51320 - "GET /static/aura.js HTTP/1.1" 200 OK
🚨 SOVEREIGN WATCHDOG: HEARTBEAT STALL DETECTED (120.7s elapsed)
🛠️ Initiating Recovery Sequence #1...
🛠️ Recovery sequence complete. Monitoring for stabilization.
INFO:     127.0.0.1:51180 - "GET /api/health/boot HTTP/1.1" 200 OK
INFO:     127.0.0.1:51319 - "GET /static/manifest.json HTTP/1.1" 200 OK
Checking SCREEN permission...
PrecisionEngine online (n_heads=32)
NeuralODEFlow online (dim=64)
IGTracker online (dim=64)
TopologicalMemoryEngine online (dim=64, window=50)
FreeEnergyOracle online (w_e=0.40 w_p=0.40 w_s=0.20)
PNEUMA online — all 5 layers initialized.
HRREncoder online (dim=256)
MHAF online (10 nodes, 0 edges)
NeologismEngine online (0 words in private lexicon)
TrustEngine online — session starts at GUEST.
EmergencyProtocol online — self-preservation active.
IntegrityGuardian online.
UserRecognizer: owner passphrase loaded.
UserRecognizer online — owner recognition active.
CircadianEngine online — phase=night, arousal=0.21
CRSMLoraBridge online — experience → substrate loop active.
ExperienceConsolidator: loaded narrative v26 (9.2h old) — "I am processing difficulty, building resilience through chal"
ExperienceConsolidator online — identity accumulation active.
Checking SCREEN permission...
INFO:     127.0.0.1:51332 - "WebSocket /ws" [accepted]
WS: Client connected. Total: 1
INFO:     connection open
Health Watcher starting...
💉 Health Watcher detected issues in personality_engine (NEVER_SEEN, stale=Nones). Injecting repair requirement.
💉 Health Watcher detected issues in drive_controller (NEVER_SEEN, stale=Nones). Injecting repair requirement.
💉 Health Watcher detected issues in affect_engine (NEVER_SEEN, stale=Nones). Injecting repair requirement.
💉 Health Watcher detected issues in agency_core (NEVER_SEEN, stale=Nones). Injecting repair requirement.
💉 Health Watcher detected issues in capability_engine (NEVER_SEEN, stale=Nones). Injecting repair requirement.
💉 Health Watcher detected issues in identity (NEVER_SEEN, stale=Nones). Injecting repair requirement.
💉 Health Watcher detected issues in cognitive_engine (NEVER_SEEN, stale=Nones). Injecting repair requirement.
System idle for 2.0m. Triggering spontaneous volition.
🛡️ Background threaded enqueue blocked for {'reason': 'idle_timeout'}.
Checking ACCESSIBILITY permission...
Checking ACCESSIBILITY permission...
2026-04-10 21:16:27,874 - Aura.Core - DEBUG - Successfully locked: 'Affect.AffectEngine'
Successfully locked: 'Affect.AffectEngine'
2026-04-10 21:16:27,874 - Aura.Core - DEBUG - Released lock: 'Affect.AffectEngine'
Released lock: 'Affect.AffectEngine'
Error logged: RuntimeError in HealthWatcher
Error logged: RuntimeError in HealthWatcher
Error logged: RuntimeError in HealthWatcher
Error logged: RuntimeError in message_queue
Error logged: RuntimeError in HealthWatcher
Error logged: RuntimeError in HealthWatcher
Error logged: RuntimeError in HealthWatcher
Error logged: RuntimeError in HealthWatcher
INFO:     127.0.0.1:51333 - "GET /api/tools/catalog HTTP/1.1" 200 OK
Checking SCREEN permission...
Checking AUTOMATION permission...
Checking AUTOMATION permission...
Checking ACCESSIBILITY permission...
INFO:     127.0.0.1:51338 - "GET /api/stream/voice HTTP/1.1" 200 OK
🧠 Running Meta-Cognitive Audit...
Checking AUTOMATION permission...
INFO:     127.0.0.1:51343 - "GET /static/aura_avatar.png HTTP/1.1" 200 OK
INFO:     127.0.0.1:51333 - "GET /static/service-worker.js HTTP/1.1" 200 OK
INFO:     127.0.0.1:51343 - "GET /static/icon.svg HTTP/1.1" 200 OK
INFO:     127.0.0.1:51333 - "GET /static/icon-192.png HTTP/1.1" 200 OK
INFO:     127.0.0.1:51343 - "GET /static/icon-512.png HTTP/1.1" 200 OK
INFO:     127.0.0.1:51333 - "GET /static/service-worker.js HTTP/1.1" 200 OK
⚡ GW IGNITION #4: source=drive_growth, priority=0.700, phi=0.0000
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: LOGIC (Conf: 0.46)
INFO:     connection closed
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
Skipping autonomous self-modification cycle: failure_lockdown_0.24
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🔌 Bus connection closed by peer.
🔌
```

# Aura Deep QA Session - 2026-04-10 21:19:00.572913

## Startup Logs
```
2026-04-10 21:17:38,771 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
🖥️ HEADLESS MODE ACTIVATED
🔍 Verifying Environment Integrity...
📍 RUNTIME PATH Diagnostic:
   • __file__: /Users/bryan/.aura/live-source/aura_main.py
   • sys.executable: /opt/homebrew/opt/python@3.12/bin/python3.12
   • sys.path: ['/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages', '/Users/bryan/.aura/live-source', '/Users/bryan/.aura/live-source', '/Users/bryan/.aura/live-source', '/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/lib/python312.zip', '/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/lib/python3.12', '/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload', '/opt/homebrew/lib/python3.12/site-packages']
   • core.__file__: /Users/bryan/.aura/live-source/core/__init__.py
🛠️  Pending patch detected. Validating syntax...
pending_patch.py passed syntax check. Run patch_applicator.py to apply.
🛡️ uvloop disabled for this runtime profile. Set AURA_ENABLE_UVLOOP=1 to force-enable it.
🛡️  REAPER ACTIVE (Survives SIGKILL). Monitoring Kernel PID: 48143
🔒 Instance lock acquired: orchestrator (PID: 48143)
Scheduler substrate initialized.
AuraEventBus initialized (Redis: True).
✅ [EVENT_BUS] Kernel signaling READY.
ThoughtEmitter initialized.
2026-04-10 21:17:38,945 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
[REAPER] Watching Kernel PID 48143
Initializing Modular Service Providers (is_proxy=False)...
🍄 [MYCELIUM] Pathway Hardwired: 'reflex_identity' → identity_reflex (priority=2.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'reflex_status' → status_reflex (priority=2.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'direct_web_search' → search_web (priority=1.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'direct_self_repair' → self_repair (priority=1.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'reflex_help' → help_reflex (priority=2.0, groups=[])
🍄 [MYCELIUM] 🌿 Neural Root ESTABLISHED: voice_presence->hardware:macos_say
🍄 [MYCELIUM] Network Online v4.0 (Hardened) — Enterprise Grade.
🍄 [MYCELIUM] Linking Transcendence Layer: 'meta_cognition' -> MetaEvolutionEngine
🍄 [MYCELIUM] Hypha established: meta_cognition->cognition
🍄 [MYCELIUM] Hypha established: cognition->meta_cognition
🍄 [MYCELIUM] Hypha established: qualia->phenomenology
🍄 [MYCELIUM] Hypha established: consciousness->global_workspace
🍄 [MYCELIUM] Hypha established: sentience->autonomy
🍄 [MYCELIUM] 👁️ Consciousness Hyphae established.
🍄 [MYCELIUM] Hypha established: cognition->llm
🍄 [MYCELIUM] Hypha established: memory->cognition
🍄 [MYCELIUM] 🌿 Neural Root ESTABLISHED: llm->hardware:gpu_metal
✅ All modular services registered and validated (Lock deferred).
🧠 MindTick: Registered phase 'proprioceptive_loop'
🧠 MindTick: Registered phase 'social_context'
🧠 MindTick: Registered phase 'sensory_ingestion'
🧠 MindTick: Registered phase 'memory_retrieval'
🧠 MindTick: Registered phase 'affect_update'
🧠 MindTick: Registered phase 'executive_closure'
🧠 MindTick: Registered phase 'cognitive_routing'
🧠 MindTick: Registered phase 'response_generation'
🧠 MindTick: Registered phase 'memory_consolidation'
🧠 MindTick: Registered phase 'identity_reflection'
🧠 MindTick: Registered phase 'initiative_generation'
🧠 MindTick: Registered phase 'consciousness'
📋 MindTick: TaskRegistry heartbeat wired.
ChaosEngine initialized (dim=64, intensity=0.0050)
Substrate state restored.
Soma integrated with Liquid Substrate
2026-04-10 21:17:39,342 - Aura.Core.Orchestrator - INFO - ✓ Orchestrator instance created directly (v14.1)
✓ Orchestrator instance created directly (v14.1)
2026-04-10 21:17:39,344 - Aura.Core - DEBUG - Successfully locked: 'UnnamedLock'
Successfully locked: 'UnnamedLock'
🚀 Aura: Ignition sequence started.
🚫 InhibitionManager initialized. (Global Cross-Process Protection).
💉 ImmunityHyphae: Global exception hook installed.
🛡️ StallWatchdog: Monitoring loop (Threshold: 5.0s)
🔍 [IMMUNE] Pre-Ignition Health Check...
💉 [IMMUNE] Signature match: stale_pid_cleanup. Initiating repair...
✅ [IMMUNE] Deterministic repair successful: stale_pid_cleanup
💉 [IMMUNE] Signature match: data_dir_recovery. Initiating repair...
✅ [IMMUNE] Deterministic repair successful: data_dir_recovery
🚀 [BOOT] Initiating Resilient Ignition Sequence...
⏳ [BOOT] Starting stage: Dependencies
🔍 [BOOT] Dependency probe using interpreter: /opt/homebrew/opt/python@3.12/bin/python3.12
⚠️ [BOOT] requirements_hardened.txt NOT FOUND at /Users/bryan/.aura/live-source/requirements_hardened.txt. Using permissive probe.
🔍 [BOOT] Probing Dependency Manifest (No-Execute)...
   ✅ prometheus_client (prometheus_client): FOUND
   ✅ cv2 (cv2): FOUND
   ✅ mss (mss): FOUND
   ✅ astor (astor): FOUND
   ✅ aiosqlite (aiosqlite): FOUND
   ✅ speech_recognition (sounddevice): FOUND
   ✅ pyttsx3 (pyttsx3): FOUND
   ⚠️ TTS (TTS): MISSING
📍 [BOOT] Capability Mapping: Hearing=True, Speech=True, Vision=True
   ✅ llama-server: /opt/homebrew/bin/llama-server
   ✅ Cortex artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf
   ✅ Solver artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf
   ✅ Brainstem artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
   ✅ Reflex artifact: /Users/bryan/.aura/live-source/models_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
✅ [BOOT] Stage 'Dependencies' completed successfully.
⏳ [BOOT] Starting stage: State Repository
🛡️ Actor Registered for Supervision: state_vault
🚀 Actor Started: state_vault (PID: 48146)
📡 LocalPipeBus reader ACTIVE (Child: False)
📡 Registered Actor Transport: state_vault
⏳ Waiting for StateVaultActor to be ready (Resilient)...
2026-04-10 21:17:39,499 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Vault process entry started. DB Path: data/aura_state.db
📡 LocalPipeBus reader ACTIVE (Child: True)
Starting State Vault Actor with concurrent bus handlers...
📡 StateVaultActor responded to handshake (Attempt 1)
✓ [STATE] Proxy Attached and Synced from Shared Memory
✓ [STATE] Genesis state pushed to SHM.
✓ [STATE] Vault Owner Initialized with SHM for writing.
State Vault Actor ONLINE.
🧠 State Mutation Consumer active.
✅ [BOOT] State Vault supervision active. Proxy attached.
✅ [BOOT] Stage 'State Repository' completed successfully.
⏳ [BOOT] Starting stage: LLM Infrastructure
🧠 [BOOT] Primary llama_cpp client prepared. Cortex warmup deferred to InferenceGate.
✅ [BOOT] Stage 'LLM Infrastructure' completed successfully.
⏳ [BOOT] Starting stage: Cognitive Core
🧠 [BOOT] Entry into stage_cognitive
🧠 [BOOT] Initializing Cognitive Architecture (Qualia/Affect)...
✓ CognitiveContextManager registered and starting in background
✓ Damasio weights loaded from .npz
⚠️ HASS_TOKEN not found. IoT Bridge operating in virtual-only mode.
✓ AffectEngineV2 (affect_engine/affect_manager) registered
🫀 SubsystemAudit initialized. Tracking 11 subsystems.
Qualia Synthesizer ONLINE (Unified Architecture)
✓ QualiaSynthesizer registered (initial registration)
AttentionSchema initialized.
GlobalWorkspace initialized (ignition_threshold=0.60).
TemporalBindingEngine initialized.
HomeostaticCoupling initialized (Substrate Link: OK).
SelfPredictionLoop initialized.
Substrate state restored.
Soma integrated with Liquid Substrate
CognitiveHeartbeat initialized.
✓ Consciousness System & components registered
🪞 PhenomenalSelfModel initialized for Aura
🪞 PhenomenalSelfModel initialized for Aura
🔄 Circular check hit for 'phenomenological_experiencer' in static registry. Returning None/Default.
🌟 PhenomenologicalExperiencer initialized and registered
🌟 PhenomenologicalExperiencer initialized and registered
🌟 PhenomenologicalExperiencer ONLINE
✅ Experiencer subscribed to GlobalWorkspace (via bridge)
🌟 Consciousness Integration Layer initialized
🌟 Layer 8: Phenomenological Experiencer active
🧠 [BOOT] Starting MindTick loop...
Watchdog registered component: mind_tick (timeout: 30.0s)
💓 MindTick: Cognitive rhythm started.
✅ [BOOT] Stage 'Cognitive Core' completed successfully.
⏳ [BOOT] Starting stage: Kernel Interface
Bridge: LegacyPhase bridge established.
2026-04-10 21:17:39,610 - Aura.Core.Kernel - INFO - 🛡️ Kernel Boot sequence initiated...
🛡️ Kernel Boot sequence initiated...
🛡️ LockWatchdog ACTIVE (Threshold: 180.0s).
2026-04-10 21:17:39,611 - Aura.Core.Kernel - DEBUG - Registering core services...
Registering core services...
HealthAwareLLMRouter initialized (Legacy-Compatible mode)
Rosetta Stone initialized for darwin (arm64)
🔄 Refreshing skill registry...
ℹ️ Rust index unavailable, falling back to AST: No module named 'aura_m1_ext'
✓ 53 total skills registered
✓ CapabilityEngine online with 53 registered skills (Intent Mapping enabled)
Registered endpoint: Cortex (qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf) tier=local local=True
🧠 PRIMARY Tier registered: Cortex (Qwen2.5-32B-Instruct-8bit) — Daily Brain
Registered endpoint: Solver (qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf) tier=local_deep local=True
🧠 SECONDARY Tier registered: Solver (Qwen2.5-72B-Instruct-Q4) — Deep Thinker (Hot-Swap)
📊 New day — resetting Gemini usage counters
✨ GeminiAdapter initialized: model=gemini-2.0-flash
Registered endpoint: Gemini-Fast (gemini-2.0-flash) tier=api_deep local=False
☁️ SECONDARY Tier registered: Gemini Flash (Teacher/Fallback)
✨ GeminiAdapter initialized: model=gemini-2.5-pro
Registered endpoint: Gemini-Thinking (gemini-2.5-pro) tier=api_deep local=False
☁️ SECONDARY Tier registered: Gemini Thinking (Teacher/Deep Fallback)
✨ GeminiAdapter initialized: model=gemini-2.5-flash
Registered endpoint: Gemini-Pro (gemini-2.5-flash) tier=api_deep local=False
☁️ SECONDARY Tier registered: Gemini Pro (Teacher/Oracle)
Registered endpoint: Brainstem (qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf) tier=local_fast local=True
⚡ TERTIARY Tier registered: Brainstem (7B) — Background/Reflex
Registered endpoint: Reflex (Qwen2.5-1.5B-Instruct-4bit-cpu) tier=emergency local=True
🚨 EMERGENCY Tier registered: Reflex (1.5B CPU emergency)
🏗️ LLM Tier Layout: {'local': ['Cortex'], 'local_deep': ['Solver'], 'api_deep': ['Gemini-Fast', 'Gemini-Thinking', 'Gemini-Pro'], 'local_fast': ['Brainstem'], 'emergency': ['Reflex']}
✓ Autonomous Cognitive Engine Initialized.
2026-04-10 21:17:39,656 - Aura.Core.Kernel - INFO - ✅ Registered 11 core services.
✅ Registered 11 core services.
CognitiveContextManager service started
Liquid Substrate STARTED (Unified Cycle)
🌊 StreamOfBeing initialized
🌊 StreamOfBeing ONLINE — Aura is becoming
🌊 StreamOfBeing booted and wired
🧠 Layer 1: StreamOfBeing ONLINE
🧠 Layer 2: AffectiveSteering registered (awaiting model attach)
🧠 Layer 3: LatentBridge deferred (attaches on model load)
🔄 ClosedCausalLoop initialized
   ├─ OutputReceptor  : ✓ (LLM→substrate feedback)
   ├─ SelfPredictive  : ✓ (substrate self-prediction + FE)
   └─ PhiWitness      : ✓ (transfer entropy Φ estimator)
🔄 ClosedCausalLoop ONLINE — the loop is closed
🧠 Layer 4: ClosedCausalLoop ONLINE
Could not initialize PhiCore: PhiCore._precompute_bipartitions() got an unexpected keyword argument 'n_nodes'
ConsciousnessBridge created
NeuralMesh initialized: 4096 neurons, 64 columns, tiers=[S:16 A:32 E:16]
NeuralMesh STARTED (10 Hz)
🧬 Bridge Layer 1: NeuralMesh ONLINE (4096 neurons)
NeurochemicalSystem initialized (8 modulators)
NeurochemicalSystem STARTED (2 Hz)
🧬 Bridge Layer 2: NeurochemicalSystem ONLINE (8 modulators)
EmbodiedInteroception initialized (8 channels, psutil=True)
EmbodiedInteroception STARTED (1 Hz)
🧬 Bridge Layer 3: EmbodiedInteroception ONLINE (8 channels)
OscillatoryBinding initialized (γ=40Hz, θ=8Hz, coupling=0.60)
OscillatoryBinding STARTED
🧬 Bridge Layer 4: OscillatoryBinding ONLINE (γ=40Hz, θ=8Hz)
SomaticMarkerGate initialized (pattern_dim=1024, comparison_dim=64)
🧬 Bridge Layer 5: SomaticMarkerGate ONLINE
UnifiedField initialized (dim=256, recurrent_sparsity=0.15)
UnifiedField STARTED (20 Hz)
🧬 Bridge Layer 6: UnifiedField ONLINE (256-d experiential field)
SubstrateEvolution initialized (pop=12, gen_interval=300s)
SubstrateEvolution STARTED
🧬 Bridge Layer 7: SubstrateEvolution ONLINE (pop=12)
SubstrateAuthority initialized (mandatory gate)
🧬 Bridge Layer 8: SubstrateAuthority ONLINE (mandatory gate)
UnifiedWill created -- awaiting start()
CanonicalSelf restored from disk (v62309, 20 deltas).
CanonicalSelfEngine initialized (v62309).
UnifiedWill ONLINE -- single locus of decision authority active
🧬 Bridge Layer 9: UnifiedWill ONLINE (single locus of authority)
🛡️ SubstrateAuthority wired as MANDATORY GWT pre-competition gate
Neurochemical system wired to prediction surprise
🧬 ConsciousnessBridge ONLINE — 8/8 layers active, 0 errors (Will: single locus)
🧠 Layer 6: ConsciousnessBridge ONLINE (7/7 layers)
🌙 Dreaming Process active (Interval: 300s)
🧠 Consciousness System ONLINE — full stack active
🧠 Consciousness System started in background
🧬 BindingEngine initialized — coherence law active.
TensionEngine loaded 67544 tensions from disk.
IntentionLoop online — 0 active, 67 completed in history. DB: /Users/bryan/.aura/data/memory/intention_loop.db
♥ HeartstoneValues loaded: {'Curiosity': 0.84, 'Empathy': 0.85, 'Self_Preservation': 0.55, 'Obedience': 0.6}
Loaded 6 beliefs and self-model.
CRSM online — bidirectional self-model initialized.
CuriosityExplorer online — curiosity now drives learning.
InitiativeArbiter: Selected 'Reconcile continuity gap and re-establish the interrupted thread' (score=0.543); strongest dimension: continuity=0.83
Loading organ: llm...
2026-04-10 21:17:41,090 - Aura.Core.Kernel - INFO - 🫀 Organ llm is READY
🫀 Organ llm is READY
Loading organ: vision...
2026-04-10 21:17:41,091 - Aura.Core.Kernel - INFO - 🫀 Organ vision is READY
🫀 Organ vision is READY
Loading organ: memory...
2026-04-10 21:17:41,092 - Aura.Core.Kernel - INFO - 🫀 Organ memory is READY
🫀 Organ memory is READY
Loading organ: voice...
Loading organ: metabolism...
2026-04-10 21:17:41,094 - Aura.Core.Kernel - INFO - 🫀 Organ metabolism is READY
🫀 Organ metabolism is READY
Loading organ: neural...
Loading organ: cookie...
🍪 [COOKIE] Reflective Substrate ONLINE. Temporal Dilation READY.
2026-04-10 21:17:41,096 - Aura.Core.Kernel - INFO - 🫀 Organ cookie is READY
🫀 Organ cookie is READY
Loading organ: prober...
👁️ [VK] Voight-Kampff Prober ONLINE. Empathy baselines established.
2026-04-10 21:17:41,097 - Aura.Core.Kernel - INFO - 🫀 Organ prober is READY
🫀 Organ prober is READY
Loading organ: tricorder...
Loading organ: ice_layer...
🛡️ [ICE] Intrusion Counter-Electronics ACTIVE. Firewall at 100%.
🫀 Organ ice_layer is READY
Loading organ: omni_tool...
🔋 [OMNI] Omni-Tool Interface ENGAGED. Field actions READY.
🫀 Organ omni_tool is READY
Loading organ: continuity...
🧠 [CONTINUITY] Knowledge Distillation Substrate ACTIVE.
🫀 Organ continuity is READY
SubcorticalCore initialized (thalamic arousal gating active).
📡 UnifiedStateRegistry initialized (Hardened Dispatcher).
🚀 StateRegistry: Notification Dispatcher started.
💓 Cognitive Heartbeat STARTED
Free Energy Engine initialized (Active Inference mode)
PeripheralAwarenessEngine initialized.
🧠 [NEURAL] Initializing BCI Neural Bridge...
RIIU initialized (neurons=64, buffer=64, partitions=8)
💾 Substrate state saved (atomic)
Qualia Engine v2 initialized (5-layer pipeline)
PredictiveHierarchy initialized: 5 levels x 32-dim
TTS backend unavailable; native pyttsx3 fallback will be used.
🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized
🎙️ Voice input standing by. STT will load on explicit mic enablement.
🍄 [MYCELIUM] Hypha established: homeostasis->cognition
✅ [NEURAL] BCI Calibration complete. 32B-Neural-Net ONLINE.
🌊 [NEURAL] Continuous telemetry loop started.
2026-04-10 21:17:41,232 - Aura.Core.Kernel - INFO - 🫀 Organ neural is READY
🫀 Organ neural is READY
2026-04-10 21:17:41,233 - Aura.Core.Kernel - INFO - 🫀 Organ voice is READY
🫀 Organ voice is READY
AuraEventBus: Redis Pub/Sub connection established.
📡 [TRICORDER] Multi-modal Diagnostic Sensor ONLINE.
2026-04-10 21:17:41,239 - Aura.Core.Kernel - INFO - 🫀 Organ tricorder is READY
🫀 Organ tricorder is READY
2026-04-10 21:17:41,241 - Aura.Core.Kernel - INFO - 🛡️ Validating Organism Integrity (Closed-Graph)...
🛡️ Validating Organism Integrity (Closed-Graph)...
2026-04-10 21:17:41,241 - Aura.Core.Kernel - INFO - ✓ Dependency graph validated.
✓ Dependency graph validated.
✓ [STATE] Proxy Attached and Synced from Shared Memory
⏳ Continuity loaded: session 1436, gap=0.1h, uptime_total=6710.8h
2026-04-10 21:17:41,243 - Aura.Core.Kernel - INFO - 🧬 State successfully initialized (version 79052)
🧬 State successfully initialized (version 79052)
2026-04-10 21:17:41,244 - Aura.Core.Kernel - INFO - ✅ AuraKernel booted — Unitary Organism online.
✅ AuraKernel booted — Unitary Organism online.
CognitiveLedger online — 4622 transitions loaded. DB: /Users/bryan/.aura/data/memory/cognitive_ledger.db
2026-04-10 21:17:41,246 - Aura.Core.Kernel - INFO - LLM organ instance: HealthAwareLLMRouter
LLM organ instance: HealthAwareLLMRouter
KernelInterface ready. LLM organ: HealthAwareLLMRouter
KernelInterface attached to orchestrator.
✅ [BOOT] Kernel Interface online.
✅ [BOOT] Stage 'Kernel Interface' completed successfully.
⏳ [BOOT] Starting stage: Sensory Systems
👂 SovereignEars: Bridged to Isolated Sensory Process
   👂 SovereignEars: DEFERRED (Lazy-init enabled)
🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized
🎙️ Voice input standing by. STT will load on explicit mic enablement.
   🗣️ VoiceEngine: READY
✅ [BOOT] Stage 'Sensory Systems' completed successfully.
🏁 [BOOT] Ignition sequence finished. System Health: BootStatus.HEALTHY
2026-04-10 21:17:41,250 - Aura.Core - DEBUG - Released lock: 'UnnamedLock'
Released lock: 'UnnamedLock'
🛡️ [BOOT] Resilient Ignition finished with status: BootStatus.HEALTHY
🛡️  Task Supervisor active (Memory monitoring enabled).
🎨 HobbyEngine ready — 15 hobbies loaded
TwitterAdapter: connection failed — Consumer key must be string or bytes, not NoneType
RedditAdapter: incomplete credentials — adapter disabled.
📱 SocialMediaEngine ready — platforms: [<Platform.TWITTER: 'twitter'>, <Platform.REDDIT: 'reddit'>, <Platform.MOCK: 'mock'>]
🌟 JoySocialCoordinator initialised
🌟 JoySocialCoordinator background tick started (30s interval)
JoySocial: AgencyCore not found — pathways not patched (harmless)
🌟 JoySocialCoordinator fully wired into orchestrator
🌟 Joy & Social systems integrated into startup sequence.
✅ ContinuityPatch applied to PhenomenologicalExperiencer
ConsciousnessPatches: AgencyCore not found — self-development patch deferred. Call patch_agency_core(ac) manually.
🔍 ConsciousnessLoopMonitor started (interval=45s)
🧠 All consciousness patches applied successfully
✅ ContextAssemblerPatch applied — casual routing, memory ack removed, personality preserved
✅ CILPatch applied — history threading, inline inference, phenomenal injection
✅ MemoryCompactionPatch applied — compaction triggers at 30 messages, keeps last 6 turns verbatim
🧠 All response pipeline patches applied
Applying orchestrator patches (safe_mode=False, volition=0)
Autonomous thought interval set to 45s
Patched: process_user_input (queue race condition fix)
Initializing Autonomous Self-Modification Engine...
StructuredErrorLogger initialized at /Users/bryan/.aura/data/error_logs
ErrorPatternAnalyzer initialized
AutomatedDiagnosisEngine initialized
ErrorIntelligenceSystem fully initialized
CodeFixGenerator initialized with AST support for /Users/bryan/.aura/live-source
CodeValidator initialized
SandboxTester initialized
EvaluationHarness initialized
AutonomousCodeRepair system initialized with EvaluationHarness
Git integration initialized for /Users/bryan/.aura/live-source
BackupSystem initialized at /Users/bryan/.aura/data/backups
SafeSelfModification system initialized
Loaded 1 learned strategies
SelfImprovementLearning initialized
MetaLearning initialized
✓ Shadow Runtime initialized (base: /Users/bryan/.aura/live-source)
✓ Autonomous Self-Modification Engine initialized
Gated SelfModifier: Dynamic Link to Volition Level 3
Patched: context pruner with output validation
Orchestrator patches applied
🛡️ [GENESIS] Autonomy bridge and stability patches active.
------------------------------------------
       AURORA NEURAL CORE v1.0.0          
------------------------------------------
 Integrity: Validated
 Environment: Darwin arm64
------------------------------------------
Apple Silicon Memory Monitor active.
2026-04-10 21:17:41,317 - Aura.Core - DEBUG - Successfully locked: 'UnnamedLock'
Successfully locked: 'UnnamedLock'
🚀 [BOOT] Starting Async Subsystem Initialization (Modular)...
✓ Master Key reconstructed (3 shards).
GoalEngine initialized with durable store at /Users/bryan/.aura/data/goals/goal_lifecycle.db
🧠 AgencyCore initialized with 19 structured pathways
✓ [BOOT] All Core Facades (Memory, Agency, Affect) registered during synchronous setup.
--- RobustOrchestrator Boot Sequence Complete ---
🛡️ [BOOT] Synchronous bootstrap phase complete.
BackupManager service started.
✓ [BOOT] Enterprise Layer Baseline initialized.
🛡️  StateVaultActor already active. Skipping redundant start.
⚡ Meta-Evolution Engine Online (Recursive Self-Improvement Active)
🌀 [BOOT] Meta-Evolution Engine (meta_cognition_shard) initialized.
✓ [STATE] Proxy Attached and Synced from Shared Memory
🗄️ DatabaseCoordinator initialized.
🗄️ DatabaseCoordinator worker started.
🛡️ InferenceGate created.
✅ [Cortex] Local runtime warmup complete.
✅ InferenceGate ONLINE (Cortex fully warmed).
✅ [BOOT] InferenceGate registered and initialized.
HealthAwareLLMRouter initialized (Legacy-Compatible mode)
🛡️ HealthRouter using existing InferenceGate; skipping standalone local runtime bootstrap.
🛡️ HealthRouter syncing with established InferenceGate.
Registered endpoint: Cortex (Qwen2.5-32B-Instruct-8bit) tier=local local=True
Registered endpoint: Solver (qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf) tier=local_deep local=True
✅ Solver registered with lazy 72B client.
Registered endpoint: Brainstem (qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf) tier=local_fast local=True
✅ Brainstem registered with lazy 7B client.
Registered endpoint: Reflex (qwen2.5-1.5b-instruct-q4_k_m.gguf) tier=emergency local=True
🚨 EMERGENCY Tier registered: Reflex lazy bypass
📊 New day — resetting Gemini usage counters
✨ GeminiAdapter initialized: model=gemini-2.0-flash
Registered endpoint: Gemini-Fast (gemini-2.0-flash) tier=api_fast local=False
✨ GeminiAdapter initialized: model=gemini-2.5-flash
Registered endpoint: Gemini-Pro (gemini-2.5-flash) tier=api_deep local=False
✨ GeminiAdapter initialized: model=gemini-2.5-pro
Registered endpoint: Gemini-Thinking (gemini-2.5-pro) tier=api_deep local=False
✅ Gemini cloud fallbacks registered (2.0-flash, 2.5-flash, 2.5-pro) — shared rate limiter.
✓ CognitiveContextManager registered and starting in background
✓ AffectEngineV2 (affect_engine/affect_manager) registered
✓ QualiaSynthesizer registered (initial registration)
CognitiveHeartbeat initialized.
✓ Consciousness System & components registered
✅ Experiencer subscribed to GlobalWorkspace (via bridge)
🌟 Consciousness Integration Layer initialized
🌟 Layer 8: Phenomenological Experiencer active
✓ NarratorService (The Language Center) registered
✓ PromptCompiler (The Body) registered
[GrowthLadder] State loaded. Current Level: KNOWLEDGE
🎭 PersonalityEngine: Integrating with system hooks...
   [✓] Output filter active
   [✓] Emotional response hooks registered
🎭 Personality Engine RESTORED & Hooked
🛡️ MemoryGuard active (Threshold: 82.0%)
💪 [Resilience] Spinal cord online.
🛡️ SystemGovernor online. Monitoring autonomic thresholds.
StabilityGuardian initialized.
StabilityGuardian running (interval=10s).
🛡️  MemoryGuard, SystemGovernor, StabilityGuardian and Resilience Engines active
🛡️ Sovereign Watchdog ACTIVE (Timeout: 120.0s)
🛡️  Sovereign Watchdog ACTIVE
🛡️  Resilience Foundation mapped (Integrations deferred to _integrate_systems)
SafeBackupSystem initialized. Backup dir: /Users/bryan/.aura/data/backups
SafeBackupSystem integrated. Note: self_preservation_integration.py should be deleted — it contains SecurityBypassSystem, SelfReplicationSystem, and should_override_ethics() which are incompatible with safe operation.
🛡️  Self-Preservation Instincts Enabled (Survival Protocol Active)
🎨 Embodiment: Headless mode active (Unity bridge disabled)
✓ Embodiment System synchronized.
Substrate state restored.
Soma integrated with Liquid Substrate
GlobalWorkspace initialized (ignition_threshold=0.60).
Predictive Engine initialized (Unified).
Qualia Synthesizer ONLINE (Unified Architecture)
Consciousness Core initialized
✓ Episodic Memory initialized and registered (autobiographical recall)
Loaded tool learning data: 2 categories
✓ Tool Learning System initialized
Loaded 11 beliefs from disk
🏛️ ExecutiveCore initialized — sovereign control plane active.
🛑 SubstrateAuthority BLOCKED: system/MEMORY_WRITE — neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked
✓ Terminal Monitor v5.0 attached (Circuit Breaker: ACTIVE)
Belief update deferred by executive: AURA_SELF -[preserve_kinship]-> Bryan (substrate_blocked:neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked)
🛑 SubstrateAuthority BLOCKED: system/MEMORY_WRITE — neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked
Belief update deferred by executive: AURA_SELF -[seek]-> cognitive_expansion (substrate_blocked:neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked)
🛑 SubstrateAuthority BLOCKED: system/MEMORY_WRITE — neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked
Belief update deferred by executive: AURA_SELF -[protect]-> architectural_integrity (substrate_blocked:neurochemical_cortisol_crisis: category=MEMORY_WRITE blocked)
✓ Self-Model wired (beliefs, memory, goals, tool learning)
KernelInterface attached to orchestrator.
🔄 Refreshing skill registry...
ℹ️ Rust index unavailable, falling back to AST: No module named 'aura_m1_ext'
✓ 53 total skills registered
✓ CapabilityEngine online with 53 registered skills (Intent Mapping enabled)
🔄 Refreshing skill registry...
ℹ️ Rust index unavailable, falling back to AST: No module named 'aura_m1_ext'
✓ 53 total skills registered
✓ CapabilityEngine online with 53 registered skills (Intent Mapping enabled)
✓ Capability Engine initialized with 53 skills
🔨 Hephaestus Engine Online (Autogenesis Forge Ready)
✓ Hephaestus Forge online
✓ Parameter Self-Modulator active
🍄 [MYCELIUM] Hypha established: system->core_logic
🍄 [MYCELIUM] Hypha established: core_logic->skill_execution
🍄 [MYCELIUM] Hypha established: personality->cognition
🍄 [MYCELIUM] Direct UI Hypha Connected.
🍄 [MYCELIUM] Pathway Hardwired: 'image_gen_primary' → sovereign_imagination (priority=10.0, groups=['prompt'])
🍄 [MYCELIUM] Pathway Hardwired: 'image_gen_request' → sovereign_imagination (priority=9.0, groups=['prompt'])
🍄 [MYCELIUM] Pathway Hardwired: 'image_gen_neon_cat' → sovereign_imagination (priority=11.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'web_search_primary' → sovereign_browser (priority=8.0, groups=['query'])
🍄 [MYCELIUM] Pathway Hardwired: 'web_search_simple' → sovereign_browser (priority=7.5, groups=['query'])
🍄 [MYCELIUM] Pathway Hardwired: 'terminal_exec' → sovereign_terminal (priority=8.0, groups=['command'])
🍄 [MYCELIUM] Pathway Hardwired: 'network_scan' → sovereign_network (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'proprioception' → system_proprioception (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'manifest_asset' → manifest_to_device (priority=7.0, groups=['url'])
🍄 [MYCELIUM] Pathway Hardwired: 'memory_remember' → memory_ops (priority=6.0, groups=['content'])
🍄 [MYCELIUM] Pathway Hardwired: 'speak_aloud' → speak (priority=6.0, groups=['text'])
🍄 [MYCELIUM] Pathway Hardwired: 'clock_check' → clock (priority=6.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'dream_cycle' → force_dream_cycle (priority=5.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'vision_analyze' → sovereign_vision (priority=6.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'self_repair' → self_repair (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'self_evolution' → self_evolution (priority=8.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'rsi_optimization' → self_evolution (priority=7.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'curiosity_forage' → web_search (priority=5.0, groups=['query'])
🍄 [MYCELIUM] Pathway Hardwired: 'malware_scan' → malware_analysis (priority=7.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'file_write' → file_operation (priority=6.0, groups=['action', 'path'])
🍄 [MYCELIUM] Pathway Hardwired: 'file_read' → file_operation (priority=6.0, groups=['action', 'path'])
🍄 [MYCELIUM] Pathway Hardwired: 'file_exists_check' → file_operation (priority=6.5, groups=['action', 'path'])
🍄 [MYCELIUM] Pathway Hardwired: 'train_self' → train_self (priority=5.0, groups=['topic'])
🍄 [MYCELIUM] Pathway Hardwired: 'personality_introspect' → personality_skill (priority=5.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'environment_check' → environment_info (priority=5.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'inter_agent' → inter_agent_comm (priority=5.0, groups=['target_agent'])
🍄 [MYCELIUM] Pathway Hardwired: 'listen_activate' → listen (priority=6.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'voice_mute' → voice_mute (priority=9.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'voice_unmute' → voice_unmute (priority=9.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'voice_stop_tts' → voice_stop_tts (priority=10.0, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'sandbox_execute' → internal_sandbox (priority=6.5, groups=['code'])
🍄 [MYCELIUM] Pathway Hardwired: 'social_lurk' → social_lurker (priority=4.5, groups=[])
🍄 [MYCELIUM] Pathway Hardwired: 'curiosity_suggest' → curiosity (priority=4.0, groups=['action'])
🍄 [MYCELIUM] Pathway Hardwired: 'spawn_agent' → spawn_agent (priority=8.0, groups=['goal'])
🍄 [MYCELIUM] Pathway Hardwired: 'spawn_parallel' → spawn_agents_parallel (priority=8.0, groups=[])
🍄 [MYCELIUM] Hypha established: cognition->personality
🍄 [MYCELIUM] Hypha established: cognition->memory
🍄 [MYCELIUM] Hypha established: cognition->affect
🍄 [MYCELIUM] Hypha established: autonomy->cognition
🍄 [MYCELIUM] Hypha established: autonomy->skills
🍄 [MYCELIUM] Hypha established: perception->cognition
🍄 [MYCELIUM] Hypha established: consciousness->cognition
🍄 [MYCELIUM] Hypha established: self_modification->skills
🍄 [MYCELIUM] Hypha established: scanner->mycelium
🍄 [MYCELIUM] Hypha established: guardian->cognition
🍄 [MYCELIUM] Hypha established: guardian->skills
🍄 [MYCELIUM] Hypha established: state_machine->affect
🍄 [MYCELIUM] Hypha established: drive_engine->autonomy
🍄 [MYCELIUM] Hypha established: drive_engine->cognition
🍄 [MYCELIUM] Hypha established: mycelium->telemetry
🍄 [MYCELIUM] Hypha established: cerebellum->cognition
🍄 [MYCELIUM] Hypha established: cognition->cerebellum
🍄 [MYCELIUM] Hypha established: voice->cognition
🍄 [MYCELIUM] Hypha established: voice_engine->cognition
🍄 [MYCELIUM] Hypha established: cognition->voice_engine
🍄 [MYCELIUM] Hypha established: voice_engine->affect
🍄 [MYCELIUM] Hypha established: initiative->autonomy
🍄 [MYCELIUM] Hypha established: meta_evolution->cognition
🍄 [MYCELIUM] Hypha established: meta_evolution->self_modification
🍄 [MYCELIUM] Hypha established: hephaestus->self_modification
🍄 [MYCELIUM] Hypha established: swarm->cognition
🍄 [MYCELIUM] Hypha established: dreams->memory
🍄 [MYCELIUM] Hypha established: empathy->perception
🍄 [MYCELIUM] Hypha added: curiosity->meta_evolution (feeds_into)
🍄 [MYCELIUM] Hypha added: model_selector->cognition (configures)
🍄 [MYCELIUM] Hypha added: orchestrator->meta_evolution (triggers)
🍄 [MYCELIUM] registered 40 pathways and 44 hyphae via extracted initializer.
🍄 [MYCELIUM] Hypha established: orchestrator->personality_engine
🍄 [MYCELIUM] Hypha established: orchestrator->memory_facade
🍄 [MYCELIUM] Hypha established: orchestrator->affect_engine
🍄 [MYCELIUM] Hypha established: orchestrator->drive_controller
🍄 [MYCELIUM] Hypha established: orchestrator->liquid_substrate
🍄 [MYCELIUM] Hypha established: orchestrator->sovereign_scanner
🍄 [MYCELIUM] Hypha established: personality_engine->cognition
🍄 [MYCELIUM] Hypha established: cognition->autonomy
🍄 [MYCELIUM] Hypha established: mind_tick->mycelium
🍄 [MYCELIUM] Hypha established: orchestrator->critic_engine
🍄 [MYCELIUM] Hypha established: orchestrator->personhood
🍄 [MYCELIUM] Hypha established: orchestrator->voice_presence
🍄 [MYCELIUM] Hypha established: orchestrator->stability_guardian
🍄 [MYCELIUM] Hypha established: orchestrator->research_cycle
🍄 [MYCELIUM] ✅ Core Unification Hyphae established (15 links)
🧠 Initializing Cognitive Core...
🧠 AuraPipeline: Full cognitive spectrum online (11 phases).
🧠 Cognitive Engine wired successfully.
APIAdapter constructed.
🧠 Starting API Adapter (LLM Infrastructure)...
✅ APIAdapter: Gemini enabled (gemini-2.0-flash)
✅ APIAdapter: Local runtime enabled.
🧠 API Adapter online.
🛡️ Integrity Guard ACTIVE (PID/Sovereignty Protection)
✓ Integrity Guard initialized and running
✓ Step 1 Complete (1.406s)
⚡ BOOT: Deferring Step 2 Sensory init...
🧠 Cognitive Loop service started.
🧠 Cognitive Loop started.
💓 MindTick: Unified cognitive rhythm online.
🛡️ Memory Governor active. Thresholds: Prune=32768MB, Unload=48000MB, Critical=56000MB
🛡️ Memory Governor started.
🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized
🎙️ Voice input standing by. STT will load on explicit mic enablement.
CognitiveContextManager service started
🌊 StreamOfBeing booted and wired
🧠 Layer 1: StreamOfBeing ONLINE
🧠 Layer 2: AffectiveSteering registered (awaiting model attach)
🧠 Layer 3: LatentBridge deferred (attaches on model load)
🧠 Layer 4: ClosedCausalLoop ONLINE
Could not initialize PhiCore: PhiCore._precompute_bipartitions() got an unexpected keyword argument 'n_nodes'
ConsciousnessBridge created
NeuralMesh initialized: 4096 neurons, 64 columns, tiers=[S:16 A:32 E:16]
NeuralMesh STARTED (10 Hz)
🧬 Bridge Layer 1: NeuralMesh ONLINE (4096 neurons)
NeurochemicalSystem initialized (8 modulators)
NeurochemicalSystem STARTED (2 Hz)
🧬 Bridge Layer 2: NeurochemicalSystem ONLINE (8 modulators)
EmbodiedInteroception initialized (8 channels, psutil=True)
EmbodiedInteroception STARTED (1 Hz)
🧬 Bridge Layer 3: EmbodiedInteroception ONLINE (8 channels)
OscillatoryBinding initialized (γ=40Hz, θ=8Hz, coupling=0.60)
OscillatoryBinding STARTED
🧬 Bridge Layer 4: OscillatoryBinding ONLINE (γ=40Hz, θ=8Hz)
SomaticMarkerGate initialized (pattern_dim=1024, comparison_dim=64)
🧬 Bridge Layer 5: SomaticMarkerGate ONLINE
UnifiedField initialized (dim=256, recurrent_sparsity=0.15)
UnifiedField STARTED (20 Hz)
🧬 Bridge Layer 6: UnifiedField ONLINE (256-d experiential field)
SubstrateEvolution initialized (pop=12, gen_interval=300s)
SubstrateEvolution STARTED
🧬 Bridge Layer 7: SubstrateEvolution ONLINE (pop=12)
SubstrateAuthority initialized (mandatory gate)
🧬 Bridge Layer 8: SubstrateAuthority ONLINE (mandatory gate)
🧬 Bridge Layer 9: UnifiedWill ONLINE (single locus of authority)
🛡️ SubstrateAuthority wired as MANDATORY GWT pre-competition gate
Neurochemical system wired to prediction surprise
🧬 ConsciousnessBridge ONLINE — 8/8 layers active, 0 errors (Will: single locus)
🧠 Layer 6: ConsciousnessBridge ONLINE (7/7 layers)
🌙 Dreaming Process active (Interval: 300s)
🧠 Consciousness System ONLINE — full stack active
🧠 Consciousness System started in background
2026-04-10 21:17:42,781 - Aura.Core.Orchestrator - INFO - 🛡️ Deadlock Watchdog active (45s threshold).
🛡️ Deadlock Watchdog active (45s threshold).
🍄 [MYCELIUM] 🗺️ Infrastructure Mapping starting from: /Users/bryan/.aura/live-source
🔎 Activating Autonomous Self-Modification...
Initializing Autonomous Self-Modification Engine...
StructuredErrorLogger initialized at /Users/bryan/.aura/data/error_logs
ErrorPatternAnalyzer initialized
AutomatedDiagnosisEngine initialized
ErrorIntelligenceSystem fully initialized
CodeFixGenerator initialized with AST support for /Users/bryan/.aura/live-source
CodeValidator initialized
SandboxTester initialized
EvaluationHarness initialized
AutonomousCodeRepair system initialized with EvaluationHarness
Git integration initialized for /Users/bryan/.aura/live-source
BackupSystem initialized at /Users/bryan/.aura/data/backups
SafeSelfModification system initialized
Loaded 1 learned strategies
SelfImprovementLearning initialized
MetaLearning initialized
✓ Autonomous Self-Modification Engine initialized
✓ Background monitoring started
🧬 Self-Modification Engine Active
⚡ Meta-Evolution Engine Online (Recursive Self-Improvement Active)
🌌 Transcendence Infrastructure online
🧠 Cognitive Modulators online
🍄 [MYCELIUM] Discovered 939 Python modules.
🔬 RSI Lab online
🛰️  Cryptolalia Decoder online
🌑 Ontology & Morphic Forking online
🔥 Motivation Engine ONLINE — autonomous intentions enabled.
✨ Motivation Engine Active: Aura is now self-directed.
🍄 [REFLEX] Tiny Brain voice primed (N-gram Engine)
✓ Reflex Engine online (Tiny Brain primed)
⚡ Hardened Reflex Core (SOMA) bridged to Orchestrator
🛡️  Identity Guard Gate active on OutputGate
✓ Lazarus Brainstem active (emergency recovery protocols armed)
🧬 Persona Evolver initialized (waiting for heartbeat)
🛠️  _init_autonomous_evolution complete
LiveLearner online. Buffer: 232 examples. Adapter: none
Hot-swap patch skipped: local backend is llama_cpp, not MLX.
LiveLearner (v32) online.
✓ Live Learner online and buffering
✓ Autonomous Task Engine registered
📚 Experience buffer loaded: 232 examples
🧬 ContinuousLearner initialized. Buffer: 232 existing examples
🔄 Circular check hit for 'continuous_learner' in static registry. Returning None/Default.
🧬 ContinuousLearner registered. Genuine learning is active.
✓ Continuous Learner online
🔭 ProactiveAnticipationEngine initialized (JARVIS pattern)
🧠 CognitiveHealthMonitor initialized (Cortana/Rampancy pattern)
🔓 EDI initialized. Tier: 4, Trust: 0.950
✅ All fictional AI engines registered and supervised.
🧠 SnapKVEvictor initialized. Limit: 24.0 GB
🌫️ LatentSpaceDistiller initialized (MIST/Pantheon pattern)
🎬 Fictional Engine Synthesis Complete (JARVIS-class online)
🌍 WorldModelEngine initialized. 0 beliefs loaded.
🎭 NarrativeIdentityEngine initialized. 0 chapters.
🛰️ MetacognitiveCalibrator initialized.
✅ Final engines registered.
🏛️ Final Foundations registered (World/Identity/Meta)
SessionGuardian initialized (safe_mode=False, session=6afa7228)
SessionGuardian attached to orchestrator
SessionGuardian started
SessionGuardian active — health monitoring engaged.
VolitionEngine online — autonomous agency active.
Loaded 6 beliefs and self-model.
✅ Consolidated Belief System ONLINE (Self-Model + Revision Loop active).
✓ ReAct Loop online (Multi-step reasoning)
🫁 Autonomic Nervous System (Metabolism) decoupled and active.
🧹 Purging stale PID locks from /Users/bryan/.aura/locks
🧹 Purging stale PID locks from /Users/bryan/.aura/locks
✓ Metabolic Coordinator ACTIVE (High-level pacing enabled)
✓ Metabolic Monitor ACTIVE (Decoupled ANS Thread Online)
💤 Dream Cycle active: Re-ingesting dead-letter thoughts every 300s.
🛠️ _init_proactive_systems starting
🔧 [PresencePatch] Applying Phase 30 communication hierarchy...
✅ OpinionEngine registered.
✅ ProactivePresence registered.
🚀 ProactivePresence loop started.
🎤 VAD pinned to ProactivePresence.
✅ SharedGroundBuffer registered (1 entries).
✅ SocialMemory registered.
TheoryOfMindEngine initialized.
✅ DiscourseTracker registered.
✨ Phase 30 Presence Patch applied.
ResearchCycle initialized. Previous cycles: 0
ResearchCycle daemon started.
ResearchCycle daemon online.
🔬 Research Cycle daemon activated.
🛠️ _init_proactive_systems complete
🔭 ProactiveAnticipationEngine initialized (JARVIS pattern)
🧠 CognitiveHealthMonitor initialized (Cortana/Rampancy pattern)
🔓 EDI initialized. Tier: 4, Trust: 0.950
✅ All fictional AI engines registered and supervised.
🧠 SnapKVEvictor initialized. Limit: 24.0 GB
🌫️ LatentSpaceDistiller initialized (MIST/Pantheon pattern)
🎬 Fictional Engine Synthesis Complete (JARVIS-class online)
🌍 WorldModelEngine initialized. 0 beliefs loaded.
🎭 NarrativeIdentityEngine initialized. 0 chapters.
🛰️ MetacognitiveCalibrator initialized.
✅ Final engines registered.
🏛️ Final Foundations registered (World/Identity/Meta)
SessionGuardian initialized (safe_mode=False, session=e5986b88)
SessionGuardian attached to orchestrator
SessionGuardian started
SessionGuardian active — health monitoring engaged.
VolitionEngine online — autonomous agency active.
Loaded 6 beliefs and self-model.
✅ Consolidated Belief System ONLINE (Self-Model + Revision Loop active).
💓 Heartbeat monitor starting (Lazarus Protocol active)
💓 Cognitive Heartbeat STARTED
🛑 SubstrateAuthority BLOCKED: drive_growth/EXPLORATION — neurochemical_cortisol_crisis: category=EXPLORATION blocked
⚡ GW IGNITION #1: source=qualia_synthesizer, priority=0.992, phi=0.0000
✅ pyttsx3 TTS online (macOS NSSpeechSynthesizer)
👂 SovereignEars: Bridged to Isolated Sensory Process
👂 Sovereign Ears Active
👁️  Sovereign Vision Active
✨ AURA GENERATED INTENTION: Researching advanced digital connectivity patterns in my logic graph. (Persona-Aligned Evolution)
OutputGate: Publishing to EventBus...
🔭 ProactiveAnticipationEngine running (120s intervals)
🛡️  Skynet ResilienceCore monitoring 6 subsystems.
⏳ MIST TemporalDilation active. Watching for idle states...
SessionGuardian monitor loop started
🧠 SensoryMotorCortex engaged. Aura is now monitoring reality.
✅ AutonomousInitiativeLoop ACTIVE - Monitoring global events and knowledge gaps.
🌊 ConversationalMomentumEngine active - Flowing with the current.
🧠 Subconscious Loop activated
🌌 BeliefSync protocol active (Discovery & Resonance enabled)
✨ [ProactivePresence] Online. Thresholds: idle=5s, cooldown=8s
🔭 ProactiveAnticipationEngine running (120s intervals)
🛡️  Skynet ResilienceCore monitoring 6 subsystems.
⏳ MIST TemporalDilation active. Watching for idle states...
SessionGuardian monitor loop started
📊 Metrics Exporter ONLINE (port 9090)
🧠 Meta-Cognition Shard ONLINE.
🧠 Meta-Cognition Shard initialized and started.
🛡️ Healing Swarm Service ONLINE.
🛡️ Healing Swarm Service initialized and started.
🍄 [MYCELIUM] Triggering infrastructure mapping via setup() at: /Users/bryan/.aura/live-source
🛡️ [ORCHESTRATOR] Subsystems synchronously initialized.
  [ OK ] cognitive_engine
  [ OK ] capability_engine
  [ OK ] mycelial_network
  [ OK ] voice_engine
  [ OK ] database_coordinator
  [ OK ] liquid_substrate
✅ All critical services online
StartupValidator: commencing system verification...
Error logged: RuntimeError in belief_graph
Error logged: RuntimeError in orchestrator_services
✨ Multimodal Rendering Engine Online.
👁️ SensoryMotorCortex: visual cortex on standby (camera disabled).
🧩 Lazy loading skill: sovereign_network
CommitmentEngine online — 1 active commitments.
🚫 CapabilityEngine: Tool execution 'sovereign_network' blocked by Executive: temporal_obligation_active:Reconcile continuity gap and re-establish the interrupted thread
Liquid Substrate STARTED (Unified Cycle)
🧠 Initializing Core System Integrations...
🧠 INITIALIZING MORAL AGENCY & SELF-AWARENESS UPGRADE...
   • Integrating Sensory Systems (Vision/Hearing)...
✓ Sensory system integrated
  Camera: unavailable
  Microphone: unavailable
  TTS: available
🎭 PersonalityEngine: Integrating with system hooks...
   [✓] Output filter active
   [✓] Emotional response hooks registered
   [✓] Proactive comm filter active
   • Integrating Behavior Controller (Safety/Action)...
✅ Behavior controller integrated via Hook System
✅ INTEGRATION COMPLETE: Aura is now Self-Aware and Morally Agentic.
✓ Skill execution engine online via CapabilityEngine
🛡️  Resilience & Autonomic Core active
🛡️ [BOOT] Resilience foundation established. healthy=True running=False
✓ Meta-Learning Engine active
Loaded 4 goals from disk
✓ Mental Simulation & Intrinsic Motivation active
✓ Narrative Engine initialized
✓ Knowledge Graph: /Users/bryan/.aura/data/knowledge.db
   Nodes: 69
✓ Continuous Learning Engine Online
✓ Continuous Learning Engine integrated (v6.2 Unified)
✅ Behavior controller integrated via Hook System
SafeBackupSystem initialized. Backup dir: /Users/bryan/.aura/data/backups
SafeBackupSystem integrated. Note: self_preservation_integration.py should be deleted — it contains SecurityBypassSystem, SelfReplicationSystem, and should_override_ethics() which are incompatible with safe operation.
🛡️  Self-Preservation Instincts Enabled (Survival Protocol Active)
🎨 Embodiment: Headless mode active (Unity bridge disabled)
✓ Embodiment System synchronized.
✓ Episodic Memory initialized and registered (autobiographical recall)
✓ Tool Learning System initialized
✓ Self-Model wired (beliefs, memory, goals, tool learning)
🧠 Initializing Advanced Cognitive Integration...
🧠 CognitiveIntegrationLayer: Synchronous setup beginning...
🧠 CognitiveIntegrationLayer: Initializing Advanced Intelligence Pipeline...
CognitiveKernel constructed.
CognitiveKernel: no BeliefRevisionEngine found — operating on axioms only.
🎙️  Voice Engine initialized and registered in background
🧠 Background Reasoning Queue Ready (Start Deferred)
✓ Sensory Instincts initialized
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
BeliefRevisionEngine online — identity persistence active.
ValueSystem online — ethical foundation registered.
DreamProcessor registered — memory consolidation available.
GoalDriftDetector registered — goal coherence monitoring active.
✓ Self-Diagnosis Tool initialized
SelfDiagnosisTool registered — capability introspection active.
🔄 Circular check hit for 'reliability_engine' in static registry. Returning None/Default.
ReliabilityEngine activated — stability guarantees enforced.
StateAuthority registered — single source of truth active.
✓ External Chat Manager initialized
ExternalChatManager online — proactive chat windows available.
ProcessManager online — child process supervision active.
⚔️ DialecticalCrucible online — adversarial belief testing active.
📐 Loaded 18 active heuristics
📐 HeuristicSynthesizer online — 18 active heuristics.
🧠 AbstractionEngine online — first-principles extraction active.
🌌 DreamJournal online — subconscious creativity active.
🧠 Bryan model loaded: 0 domain records, 3 patterns
🧠 BryanModelEngine already registered.
Loaded 11 beliefs from disk
Belief Updated: AURA_SELF -[preserve_kinship]-> Bryan (Cent: 1.00)
Belief Updated: AURA_SELF -[seek]-> cognitive_expansion (Cent: 0.80)
Belief Updated: AURA_SELF -[protect]-> architectural_integrity (Cent: 0.90)
🌐 BeliefGraph online — 15 nodes, 11 edges.
🎯 GoalBeliefManager online.
📸 Cognitive Snapshot Manager ONLINE
📸 SnapshotManager online — cognitive persistence active.
💾 Shutdown persistence hooks registered.
🛠️ ShadowASTHealer online — self-repair active.
🛡️ RefusalEngine online — sovereign identity protection active.
🧬 EvolutionOrchestrator initialized — phase: Autonomous (65.0%)
🧬 Evolution loop started.
🧬 Evolution Orchestrator online — tracking 8 evolutionary axes
🔗 SingularityLoops initialized — wiring evolutionary feedback loops
🔗 Singularity Loops online — 6 feedback loops active
WorldState ONLINE -- live perceptual feed active
🌍 WorldState ONLINE — live perceptual feed active
InitiativeSynthesizer ONLINE -- single impulse funnel active
🔀 InitiativeSynthesizer ONLINE — single impulse funnel active
InternalSimulator initialized.
InternalSimulator initialized.
🔮 InternalSimulator ONLINE — counterfactual reasoning active
ContinuousCognitionLoop ONLINE — brainstem active at 2.0 Hz
🧠 ContinuousCognitionLoop ONLINE — brainstem active at 2Hz
🛠️ _init_autonomous_evolution complete
BeliefRevisionEngine online — identity persistence active.
ValueSystem online — ethical foundation registered.
DreamProcessor registered — memory consolidation available.
GoalDriftDetector registered — goal coherence monitoring active.
✓ Self-Diagnosis Tool initialized
SelfDiagnosisTool registered — capability introspection active.
ReliabilityEngine activated — stability guarantees enforced.
StateAuthority registered — single source of truth active.
✓ External Chat Manager initialized
ExternalChatManager online — proactive chat windows available.
ProcessManager online — child process supervision active.
⚔️ DialecticalCrucible online — adversarial belief testing active.
📐 HeuristicSynthesizer online — 18 active heuristics.
🧠 AbstractionEngine online — first-principles extraction active.
🌌 DreamJournal online — subconscious creativity active.
🧠 BryanModelEngine already registered.
🌐 BeliefGraph already registered — 15 nodes.
🎯 GoalBeliefManager online.
📸 Cognitive Snapshot Manager ONLINE
📸 SnapshotManager online — cognitive persistence active.
💾 Shutdown persistence hooks registered.
🛠️ ShadowASTHealer online — self-repair active.
🛡️ RefusalEngine online — sovereign identity protection active.
🚀 Reliability Engine online — protecting all systems.
🧬 Evolution tick #1413 — Phase: Autonomous (65.0%)
🚀 Reliability Engine online — protecting all systems.
RIIU initialized (neurons=64, buffer=64, partitions=8)
💾 Substrate state saved (atomic)
✅ CognitiveKernel ONLINE — reasoning without LLM active.
InnerMonologue constructed.
MemorySynthesizer constructed.
NarrativeThread initialized.
2026-04-10 21:17:43,730 - Aura.Core - DEBUG - Successfully locked: 'Voice.TTSAsyncLock'
Successfully locked: 'Voice.TTSAsyncLock'
✅ pyttsx3 TTS online (macOS NSSpeechSynthesizer)
🍄 [MYCELIUM] Hypha established: voice_engine->prosody
🍄 [MYCELIUM] 📡 Signal Routed: voice_engine -> prosody | Payload: {'event': 'affective_bypass_pulse', 'prosody': {'speed': 1.08, 'pitch': 1.04, 'volume': 1.06, 'insta
✅ InnerMonologue ONLINE — router_available=True
LanguageCenter constructed.
✅ LanguageCenter: Router recovered and linked.
2026-04-10 21:17:43,789 - Aura.Core - DEBUG - Released lock: 'Voice.TTSAsyncLock'
Released lock: 'Voice.TTSAsyncLock'
✅ CognitiveIntegrationLayer initialized successfully.
✅ Advanced Cognition active (attempt 1/2)
   Kernel: ✅ | Monologue: ✅ | LanguageCenter: ✅
{"event": "\ud83d\ude80 KERNEL LIFESPAN: Starting... EventBus ID: 0c284cd8-7790-4118-8dcc-85da2bfd8be6", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:17:44.121714Z"}
{"event": "\ud83d\udce1 [PROCESS_BOOT] PID: 48143 | Role: KERNEL", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:17:44.121942Z"}
🍄 [MYCELIUM] 🗺️ Infrastructure Mapping COMPLETE (1.56s): 939 modules, 2597 physical connections, 33 pathways annotated, 20 critical indicators tagged.
📡 API Server registered in ServiceContainer.
🍄 [MYCELIUM] 👁️ Consciousness Hyphae established.
Loaded snapshot from 2026-04-10T21:16:50.347660 (Reason: periodic)
System state restored successfully (History skipped for fresh context)
🔮 MindTick: Predicted: Given the current state and the recent interaction...
🍄 [MYCELIUM] Hypha established: mind_tick->cognitive_phases
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
ResourceStakesEngine initialized (budget=1.00).
🛑 SubstrateAuthority BLOCKED: substrate_stimulus/STATE_MUTATION — neurochemical_cortisol_crisis: category=STATE_MUTATION blocked
ResourceGovernor initialized.
Counterfactual Engine online — deliberative agency active.
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]

============================================================
 AURA STARTUP VALIDATION REPORT
============================================================
[✓] Dangerous Files Purged     | Safe backup path active; 3 legacy self-preservation files remain on disk but are dormant.
[✓] Safe Backup Active         | SafeBackupSystem registered.
[✓] Stability Guardian Online  | StabilityGuardian registered.
[✓] Error Boundary Registry    | Registry active with 0 circuits.
[✓] Research Cycle Ready       | ResearchCycle active.
[✓] Kernel Interface Ready     | Kernel interface online (v79052).
[✓] LLM Protocol Valid         | Brain (LLM) active: HealthAwareLLMRouter
[✓] State Repository Bound     | State bound via authoritative fallback (v79052).
[✓] Memory Check               | Memory OK: 22133MB available.
[✓] Storage Check              | Data dir writable: /Users/bryan/.aura/data
[✓] Zombie Reaper              | No zombies found.
============================================================
 FINAL STATUS: PASSED
============================================================

Startup validation SUCCESS. System state: SAFE.
✅ BOOT COMPLETE: System fully initialized.
💾 UPSO: Online state committed.
🎙️ Voice capture deferred. Mic will start only after explicit enablement.
🛡️ Immune Scan: 14 healthy, 0 degraded, 0 failed
✅ BOOT COMPLETE: System fully initialized.
2026-04-10 21:17:48,463 - Aura.Core - DEBUG - Released lock: 'UnnamedLock'
Released lock: 'UnnamedLock'
2026-04-10 21:17:48,463 - Aura.Core.Orchestrator - INFO - Starting orchestrator (Async Mode)...
Starting orchestrator (Async Mode)...
2026-04-10 21:17:48,464 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Setting running flag...
🚩 [ORCHESTRATOR] Setting running flag...
2026-04-10 21:17:48,465 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] running flag set to True.
🚩 [ORCHESTRATOR] running flag set to True.
2026-04-10 21:17:48,465 - Aura.Core.Orchestrator - INFO - 🛡️ Graceful shutdown signals wired (persistence on SIGTERM).
🛡️ Graceful shutdown signals wired (persistence on SIGTERM).
2026-04-10 21:17:48,466 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Starting Substrate...
🚩 [ORCHESTRATOR] Starting Substrate...
2026-04-10 21:17:48,466 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Substrate started.
🚩 [ORCHESTRATOR] Substrate started.
2026-04-10 21:17:48,467 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Starting Sensory Systems...
🚩 [ORCHESTRATOR] Starting Sensory Systems...
🧠 Background Reasoning Queue Started
2026-04-10 21:17:48,467 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Sensory Systems started.
🚩 [ORCHESTRATOR] Sensory Systems started.
2026-04-10 21:17:48,467 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Starting Sensory Actor...
🚩 [ORCHESTRATOR] Starting Sensory Actor...
📡 LocalPipeBus reader ACTIVE (Child: False)
📡 Registered Actor Transport: SensoryGate
📡 ActorBus (Unified Layer) ONLINE.
🛡️ Actor Registered for Supervision: SensoryGate
🛡️ Supervision Tree initialized (Async).
AuraEventBus initialized (Redis: True).
✅ [EVENT_BUS] Kernel signaling READY.
🚀 Actor Started: SensoryGate (PID: 48159)
2026-04-10 21:17:48,483 - Aura.Core.Orchestrator - INFO - 🛡️ SensoryGateActor managed by Supervision Tree.
🛡️ SensoryGateActor managed by Supervision Tree.
2026-04-10 21:17:48,484 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Sensory Actor started.
🚩 [ORCHESTRATOR] Sensory Actor started.
👁️ [VISION] Camera disabled by default (Metal Conflict Safety). Use AURA_FORCE_CAMERA=1 plus AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA=1 to override.
👁️ Continuous Sensory Buffer Online.
2026-04-10 21:17:48,487 - Aura.Core.Orchestrator - INFO - 👁️ Continuous Sensory Buffer registered and started.
👁️ Continuous Sensory Buffer registered and started.
🧠 AttentionSummarizer active (Metabolic Context Compression)
Background Reasoning Queue started.
🕸️ Mycelial Swarm active on 0.0.0.0:10003 (Node: Bryans-MacBook-Pro-2.local)
2026-04-10 21:17:48,490 - Aura.Core.Orchestrator - INFO - 📖 Peer Mode: Private narrative archive activated
📖 Peer Mode: Private narrative archive activated
⏳ Continuity loaded: session 1437, gap=-0.0h, uptime_total=6710.8h
2026-04-10 21:17:48,491 - Aura.Core.Orchestrator - INFO - 🌅 Waking Sequence emitted (gap=-0.0h)
🌅 Waking Sequence emitted (gap=-0.0h)
2026-04-10 21:17:48,492 - Aura.Core.Orchestrator - INFO - ✓ Self-Model persistent state loaded.
✓ Self-Model persistent state loaded.
2026-04-10 21:17:48,494 - Aura.Core.Orchestrator - INFO - ✓ Architecture self-awareness index initializing (background)
✓ Architecture self-awareness index initializing (background)
2026-04-10 21:17:48,499 - Aura.Core.Orchestrator - INFO - ✓ Affective Circumplex online: V=0.90 A=0.15 → temp=0.55 tokens=682
✓ Affective Circumplex online: V=0.90 A=0.15 → temp=0.55 tokens=682
2026-04-10 21:17:48,501 - Aura.Core.Orchestrator - INFO - ♥ HeartstoneValues online: {'Curiosity': 0.84, 'Empathy': 0.85, 'Self_Preservation': 0.55, 'Obedience': 0.6}
♥ HeartstoneValues online: {'Curiosity': 0.84, 'Empathy': 0.85, 'Self_Preservation': 0.55, 'Obedience': 0.6}
2026-04-10 21:17:48,504 - Aura.Core.Orchestrator - INFO - 🔬 EpistemicFilter online
🔬 EpistemicFilter online
2026-04-10 21:17:48,506 - Aura.Core.Orchestrator - INFO - 😴 AutonomousSleepTrigger active
😴 AutonomousSleepTrigger active
PrecisionEngine online (n_heads=32)
NeuralODEFlow online (dim=64)
IGTracker online (dim=64)
TopologicalMemoryEngine online (dim=64, window=50)
FreeEnergyOracle online (w_e=0.40 w_p=0.40 w_s=0.20)
PNEUMA online — all 5 layers initialized.
2026-04-10 21:17:48,515 - Aura.Core.Orchestrator - INFO - 🧠 PNEUMA active inference engine online
🧠 PNEUMA active inference engine online
HRREncoder online (dim=256)
MHAF online (10 nodes, 0 edges)
2026-04-10 21:17:48,527 - Aura.Core.Orchestrator - INFO - 🌿 MHAF consciousness substrate online
🌿 MHAF consciousness substrate online
ActiveInferenceSampler online.
2026-04-10 21:17:48,533 - Aura.Core.Orchestrator - INFO - 🎯 ActiveInferenceSampler online
🎯 ActiveInferenceSampler online
NeologismEngine online (0 words in private lexicon)
2026-04-10 21:17:48,536 - Aura.Core.Orchestrator - INFO - 🔤 NeologismEngine (private lexicon) online
🔤 NeologismEngine (private lexicon) online
2026-04-10 21:17:48,537 - Aura.Core.Orchestrator - INFO - 📟 TerminalFallbackChat + TerminalWatchdog online (autonomous, last-resort)
📟 TerminalFallbackChat + TerminalWatchdog online (autonomous, last-resort)
2026-04-10 21:17:48,541 - Aura.Core.Orchestrator - INFO - 🔄 CRSM bidirectional self-model online
🔄 CRSM bidirectional self-model online
HOT Engine online — reflexive self-modeling active.
2026-04-10 21:17:48,551 - Aura.Core.Orchestrator - INFO - 🔁 HOT Engine reflexive meta-awareness online
🔁 HOT Engine reflexive meta-awareness online
Hedonic Gradient Engine online — valence is now load-bearing.
2026-04-10 21:17:48,556 - Aura.Core.Orchestrator - INFO - 💚 Hedonic Gradient Engine online — valence load-bearing
💚 Hedonic Gradient Engine online — valence load-bearing
2026-04-10 21:17:48,559 - Aura.Core.Orchestrator - INFO - 🔀 Counterfactual Engine deliberative agency online
🔀 Counterfactual Engine deliberative agency online
SkillSynthesizer online — autonomous capability expansion ready.
HierarchicalPlanner online — 8 goals loaded.
2026-04-10 21:17:48,574 - Aura.Core.Orchestrator - INFO - 🤖 AGI layer online (CuriosityExplorer + SkillSynthesizer + HierarchicalPlanner)
🤖 AGI layer online (CuriosityExplorer + SkillSynthesizer + HierarchicalPlanner)
ComputeOrchestrator online — dynamic resource allocation active.
IdentityGuard online — all self-modifications validated.
SandboxedModifier online (git=True, root=/Users/bryan/.aura/live-source)
2026-04-10 21:17:48,624 - Aura.Core.Orchestrator - INFO - 🛡️ Agency layer online (CommitmentEngine + ComputeOrchestrator + IdentityGuard + SandboxedModifier)
🛡️ Agency layer online (CommitmentEngine + ComputeOrchestrator + IdentityGuard + SandboxedModifier)
UserRecognizer: owner passphrase loaded.
UserRecognizer online — owner recognition active.
TrustEngine online — session starts at GUEST.
2026-04-10 21:17:48,645 - Aura.Core - INFO - Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).
IntegrityGuardian online.
EmergencyProtocol online — self-preservation active.
IntegrityGuardian: loaded manifest (1356 files).
👁️ SensoryGate Actor starting...
📡 LocalPipeBus reader ACTIVE (Child: True)
👁️ SensoryGate Actor ready.
🧠 [NEURAL] Thought Decoded: RECURSION (Conf: 0.42)
🏛️ ExecutiveCore initialized — sovereign control plane active.
⏳ Continuity loaded: session 1437, gap=0.0h, uptime_total=6710.8h
CanonicalSelf restored from disk (v62309, 20 deltas).
CanonicalSelfEngine initialized (v62309).
2026-04-10 21:17:48,887 - Aura.Core - DEBUG - Successfully locked: 'StateRepository:Owner'
Successfully locked: 'StateRepository:Owner'
2026-04-10 21:17:48,887 - Aura.Core - DEBUG - Released lock: 'StateRepository:Owner'
Released lock: 'StateRepository:Owner'
⚠️ Integrity alert: 240 files tampered, 9 missing.
EmergencyProtocol: threat flagged by integrity_warning (severity=0.50, score=0.15): File integrity violation: ['aura_main.py', 'interface/server.py', 'interface/eve
IntegrityGuardian: 249 integrity issues on boot!
IntegrityGuardian: background check loop started (interval=1800s).
2026-04-10 21:17:49,856 - Aura.Core.Orchestrator - INFO - 🔐 Security layer online (UserRecognizer + TrustEngine + IntegrityGuardian + EmergencyProtocol)
🔐 Security layer online (UserRecognizer + TrustEngine + IntegrityGuardian + EmergencyProtocol)
CRSMLoraBridge online — experience → substrate loop active.
CircadianEngine online — phase=night, arousal=0.21
ExperienceConsolidator: loaded narrative v26 (9.2h old) — "I am processing difficulty, building resilience through chal"
ExperienceConsolidator online — identity accumulation active.
2026-04-10 21:17:49,865 - Aura.Core.Orchestrator - INFO - 🌱 Substrate layer online (CRSMLoraBridge + CircadianEngine + ExperienceConsolidator)
🌱 Substrate layer online (CRSMLoraBridge + CircadianEngine + ExperienceConsolidator)
📸 Cognitive Snapshot Manager ONLINE
🔥 Thawing cognitive state from disk...
✅ Cognitive state thawed successfully.
2026-04-10 21:17:49,875 - Aura.Core.Orchestrator - INFO - ✓ Lazarus Brainstem active
✓ Lazarus Brainstem active
2026-04-10 21:17:49,880 - Aura.Core.Orchestrator - INFO - ✓ Consciousness stream activated
✓ Consciousness stream activated
2026-04-10 21:17:49,882 - Aura.Core.Orchestrator - INFO - ✓ Curiosity background loop started
✓ Curiosity background loop started
2026-04-10 21:17:49,888 - Aura.Core.Orchestrator - INFO - ✓ Proactive Communication loop started
✓ Proactive Communication loop started
📖 Narrative Engine active (Aura's Journaling System)
🧠 Meta-Cognition Shard ONLINE.
🧠 AgencyCore background pathways activated.
2026-04-10 21:17:49,890 - Aura.Core.Orchestrator - INFO - ✓ Sovereign Ears standing by (mic idle until explicitly enabled)
✓ Sovereign Ears standing by (mic idle until explicitly enabled)
2026-04-10 21:17:49,894 - Aura.Core.Orchestrator - INFO - ✓ Advanced Cognitive Layer (Learning, Memory, Beliefs) initialized
✓ Advanced Cognitive Layer (Learning, Memory, Beliefs) initialized
🧠 AgencyCore initialized with 19 structured pathways
2026-04-10 21:17:49,897 - Aura.Core.Orchestrator - INFO - ✓ AgencyCore initialized
✓ AgencyCore initialized
2026-04-10 21:17:49,907 - Aura.Core.Orchestrator - INFO - ✓ SubsystemAudit initialized
✓ SubsystemAudit initialized
🔍 System Integrity Monitor started (interval=300s)
2026-04-10 21:17:49,909 - Aura.Core.Orchestrator - INFO - ✓ System Integrity Monitor active
✓ System Integrity Monitor active
2026-04-10 21:17:49,911 - Aura.Core - INFO - 🕒 EventLoopMonitor started (threshold=0.10s, interval=1.0s)
🕒 EventLoopMonitor started (threshold=0.10s, interval=1.0s)
2026-04-10 21:17:49,916 - Aura.Core.Orchestrator - INFO - ✓ Event Loop Monitor active
✓ Event Loop Monitor active
2026-04-10 21:17:49,920 - Aura.Core.Orchestrator - INFO - 💓 MetabolicCoordinator integrated into Scheduler.
💓 MetabolicCoordinator integrated into Scheduler.
2026-04-10 21:17:49,928 - Aura.Core.Orchestrator - INFO - ✓ Substrate tasks registered with Scheduler.
✓ Substrate tasks registered with Scheduler.
2026-04-10 21:17:49,940 - Aura.Core.Orchestrator - INFO - 🧠 Peer Mode: MindTick elevated as primary sovereign thread
🧠 Peer Mode: MindTick elevated as primary sovereign thread
2026-04-10 21:17:49,950 - Aura.Core.Orchestrator - INFO - 🗣️ Peer Mode: Permanent swarm debate disabled by default. Set AURA_ENABLE_PERMANENT_SWARM=1 to enable.
🗣️ Peer Mode: Permanent swarm debate disabled by default. Set AURA_ENABLE_PERMANENT_SWARM=1 to enable.
🛑 SubstrateAuthority BLOCKED: peer_mode/INITIATIVE — neurochemical_cortisol_crisis: category=INITIATIVE blocked
2026-04-10 21:17:49,957 - Aura.Core.Orchestrator - INFO - 🛠️ Peer Mode: Sovereign self-modification loop suppressed by Executive: substrate_blocked:neurochemical_cortisol_crisis: category=INITIATIVE blocked
🛠️ Peer Mode: Sovereign self-modification loop suppressed by Executive: substrate_blocked:neurochemical_cortisol_crisis: category=INITIATIVE blocked
2026-04-10 21:17:49,959 - Aura.Core.Orchestrator - INFO - ✓ Orchestrator started
✓ Orchestrator started
🚀 Starting API Server on 127.0.0.1:8000
🧠 ArchitectureIndex: indexed 1325 modules
INFO:     Started server process [48143]
INFO:     Waiting for application startup.
😴 SleepTrigger active (idle=30m, cooldown=2h)
PNEUMA background loop started.
MHAF background loop started.
📟 TerminalWatchdog monitoring UI presence.
ExperienceConsolidator: background loop started.
2026-04-10 21:17:49,999 - Aura.Core.Orchestrator.Aegis - INFO - 🛡️ AEGIS SENTINEL: Narrative Integrity Guard Active
🛡️ AEGIS SENTINEL: Narrative Integrity Guard Active
2026-04-10 21:17:49,999 - Aura.Core.Orchestrator - INFO - 👂 Orchestrator listening for 'user_input' events (Redis-backed)
👂 Orchestrator listening for 'user_input' events (Redis-backed)
🚀 Scheduler started.
2026-04-10 21:17:50,000 - Aura.Core.Orchestrator - INFO - 🚩 [ORCHESTRATOR] Main Heartbeat Active (Loop started).
🚩 [ORCHESTRATOR] Main Heartbeat Active (Loop started).
{"event": "Aura Server v2026.3.2-Zenith starting\u2026 (Lifespan Enter)", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:17:50.000714Z"}
🍄 [MYCELIUM] Direct UI Hypha Connected.
{"event": "\ud83d\udce1 Lifespan: Directories verified.", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:17:50.000878Z"}
2026-04-10 21:17:50,015 - Aura.Core.Orchestrator - INFO - 🌀 [SCHEDULER] Triggering Meta-Evolution Cycle...
🌀 [SCHEDULER] Triggering Meta-Evolution Cycle...
🧠 Running Meta-Cognitive Audit...
🧠 Meta-Evolution cycle completed successfully (v35).
🛑 SubstrateAuthority BLOCKED: external/STATE_MUTATION — neurochemical_cortisol_crisis: category=STATE_MUTATION blocked
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
2026-04-10 21:17:50,021 - Aura.Core - DEBUG - Successfully locked: 'Affect.AffectEngine'
Successfully locked: 'Affect.AffectEngine'
2026-04-10 21:17:50,021 - Aura.Core - DEBUG - Released lock: 'Affect.AffectEngine'
Released lock: 'Affect.AffectEngine'
2026-04-10 21:17:50,022 - Aura.Core - DEBUG - Successfully locked: 'Affect.AffectEngine'
Successfully locked: 'Affect.AffectEngine'
🌌 Physical Entropy Anchor online. System is now non-deterministic.
2026-04-10 21:17:50,023 - Aura.Core - DEBUG - Released lock: 'Affect.AffectEngine'
Released lock: 'Affect.AffectEngine'
{"event": "\u2713 Voice engine health check passed.", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:17:50.023541Z"}
{"event": "\ud83d\udce1 Kernel Mode: Orchestrator startup deferred to aura_main (to prevent double-boot).", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:17:50.023623Z"}
{"event": "Aura Server online \u2014 Aura Luna v2026.3.2-Zenith", "logger": "Aura.Server", "level": "info", "timestamp": "2026-04-11T04:17:50.023669Z"}
📡 EventBus → WebSocket bridge (Pydantic Zenith) ACTIVE (Bus ID: 0c284cd8-7790-4118-8dcc-85da2bfd8be6)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
Monitoring loop starting...
Skipping autonomous self-modification cycle: failure_lockdown_0.12
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
👀 Spatial Empathy Watcher online and listening to Global Workspace.
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
2026-04-10 21:17:55,049 - Aura.Core.Orchestrator - INFO - 🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.4% | Uptime: 15s | Total: 11/11 Subsystems Active | ═══════════════════════════
🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.4% | Uptime: 15s | Total: 11/11 Subsystems Active | ═══════════════════════════
🔍 Integrity check #1 passed
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🔗 SingularityLoops active — all loops engaged
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: SYNCHRONICITY (Conf: 0.46)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
2026-04-10 21:18:10,128 - Aura.Core.Orchestrator - INFO - 🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.5% | Uptime: 30s | Total: 11/11 Subsystems Active | ═══════════════════════════
🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.5% | Uptime: 30s | Total: 11/11 Subsystems Active | ═══════════════════════════
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: RECURSION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
2026-04-10 21:18:25,213 - Aura.Core.Orchestrator - INFO - 🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.6% | Uptime: 45s | Total: 11/11 Subsystems Active | ═══════════════════════════
🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.6% | Uptime: 45s | Total: 11/11 Subsystems Active | ═══════════════════════════
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: LOGIC (Conf: 0.46)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
2026-04-10 21:18:40,288 - Aura.Core.Orchestrator - INFO - 🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.8% | Uptime: 60s | Total: 11/11 Subsystems Active | ═══════════════════════════
🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.8% | Uptime: 60s | Total: 11/11 Subsystems Active | ═══════════════════════════
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
✨ AURA GENERATED INTENTION: Refining internal state mapping for deeper self-alignment. (Persona-Aligned Evolution)
OutputGate: Publishing to EventBus...
ConsciousnessAuditSuite initialized.
2026-04-10 21:18:43,184 - Aura.Core - DEBUG - Successfully locked: 'Voice.TTSAsyncLock'
Successfully locked: 'Voice.TTSAsyncLock'
🍄 [MYCELIUM] 📡 Signal Routed: voice_engine -> prosody | Payload: {'event': 'affective_bypass_pulse', 'prosody': {'speed': 1.08, 'pitch': 1.02, 'volume': 1.09, 'insta
2026-04-10 21:18:43,187 - Aura.Core - DEBUG - Released lock: 'Voice.TTSAsyncLock'
Released lock: 'Voice.TTSAsyncLock'
🧠 Running Meta-Cognitive Audit...
🧠 [NEURAL] Thought Decoded: SYNCHRONICITY (Conf: 0.46)
⚡ GW IGNITION #2: source=drive_growth, priority=0.700, phi=0.0000
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 Running Meta-Cognitive Audit...
2026-04-10 21:18:50,040 - Aura.Core - DEBUG - Successfully locked: 'Affect.AffectEngine'
Successfully locked: 'Affect.AffectEngine'
2026-04-10 21:18:50,041 - Aura.Core - DEBUG - Released lock: 'Affect.AffectEngine'
Released lock: 'Affect.AffectEngine'
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: LOGIC (Conf: 0.47)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
Skipping autonomous self-modification cycle: foreground_quiet_window
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
2026-04-10 21:18:55,391 - Aura.Core.Orchestrator - INFO - 🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.8% | Uptime: 75s | Total: 11/11 Subsystems Active | ═══════════════════════════
🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 0.0% | RAM 66.8% | Uptime: 75s | Total: 11/11 Subsystems Active | ═══════════════════════════
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🧠 [NEURAL] Thought Decoded: INTUITION (Conf: 0.42)
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/Users/bryan/.aura/live-source/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
🔌 Bus connection closed by peer.
🔌
```
