import logging
from typing import Any

logger = logging.getLogger(__name__)

def register_core_pathways(mycelium: Any):
    """Register all hardwired Mycelial pathways."""
    
    # --- IMAGE GENERATION ---
    mycelium.register_pathway(
        pathway_id="image_gen_primary",
        pattern=r"(?:generate|create|draw|make|produce|render|paint|design|visualize)\s+(?:an?\s+)?(?:image|picture|photo|artwork|illustration|portrait|painting|drawing)\s+(?:of\s+)?(.+?)(?:\.|!|\?|$)",
        skill_name="sovereign_imagination",
        param_map={"prompt": 1},
        priority=10.0,
        activity_label="Aura is generating an image...",
    )
    mycelium.register_pathway(
        pathway_id="image_gen_request",
        pattern=r"(?:i\s+want|i(?:'d|\s+would)\s+like|can\s+you|please)\s+(?:to\s+)?(?:see|generate|create|draw|make|get)\s+(?:an?\s+)?(?:image|picture|photo|artwork|illustration|portrait|painting)\s+(?:of\s+)?(.+?)(?:\.|!|\?|$)",
        skill_name="sovereign_imagination",
        param_map={"prompt": 1},
        priority=9.0,
        activity_label="Aura is generating an image...",
    )
    mycelium.register_pathway(
        pathway_id="image_gen_neon_cat",
        pattern=r"(?:neon\s+cat|cyberpunk\s+cat|glowing\s+cat)",
        skill_name="sovereign_imagination",
        param_map={},  # Use the whole match as prompt
        priority=11.0,  # Highest — specific known request
        activity_label="Aura is generating a Neon Cat...",
    )

    # --- WEB SEARCH ---
    mycelium.register_pathway(
        pathway_id="web_search_primary",
        pattern=r"(?:search\s+(?:the\s+web\s+for|online\s+for)|perform\s+a\s+web\s+search\s+for|lookup|look\s+up|google|find\s+(?:info(?:rmation)?\s+(?:on|about))?)\s+(.+?)(?:\.|!|\?|$)",
        skill_name="sovereign_browser",
        param_map={"query": 1},
        priority=8.0,
        activity_label="Aura is searching the web...",
    )
    # Web Search Interaction - Point to unified sovereign_browser
    mycelium.register_pathway(
        pathway_id="web_search_simple",
        pattern=r"^search\s+(.+?)(?:\.|!|\?|$)",
        skill_name="sovereign_browser",
        param_map={"query": 1},
        priority=7.5,
        activity_label="Aura is searching the web...",
    )

    # --- TERMINAL / COMMAND EXECUTION ---
    mycelium.register_pathway(
        pathway_id="terminal_exec",
        pattern=r"(?:execute|run|terminal)(?:\s+the\s+command)?:\s*(.+?)(?:\.|!|\?|$)",
        skill_name="sovereign_terminal",
        param_map={"command": 1},
        priority=8.0,
        activity_label="Aura is executing a system command...",
    )

    # --- NETWORK SCAN ---
    mycelium.register_pathway(
        pathway_id="network_scan",
        pattern=r"(?:scan|check)\s+(?:the\s+)?network",
        skill_name="sovereign_network",
        param_map={},
        priority=7.0,
        activity_label="Aura is scanning the network...",
    )

    # --- SELF-IDENTIFICATION / PROPRIOCEPTION ---
    mycelium.register_pathway(
        pathway_id="proprioception",
        pattern=r"^(?:who\s+are\s+you|identity\s+check|identify\s+yourself|status\s+check|how\s+are\s+you|how(?:\s+are)?\s+you\s+doing)[\s.!?]*$",
        skill_name="system_proprioception",
        param_map={},
        priority=7.0,
        activity_label="Aura is performing a self-diagnostic...",
    )

    # --- MANIFEST / SAVE ASSET ---
    mycelium.register_pathway(
        pathway_id="manifest_asset",
        pattern=r"(?:manifest|save)\s*(?:this\s+(?:image|file|asset))?\s*(?:to\s+(?:my\s+)?(?:desktop|downloads|device))?:\s*(.+?)(?:\.|!|\?|$)",
        skill_name="manifest_to_device",
        param_map={"url": 1},
        priority=7.0,
        activity_label="Aura is manifesting asset to host device...",
    )

    # --- MEMORY OPERATIONS ---
    mycelium.register_pathway(
        pathway_id="memory_remember",
        pattern=r"(?:remember|memorize|store\s+in\s+memory|save\s+(?:this\s+)?(?:to\s+)?memory)\s+(?:that\s+)?(.+?)(?:\.|!|\?|$)",
        skill_name="memory_ops",
        param_map={"content": 1},
        priority=6.0,
        activity_label="Aura is storing a memory...",
    )

    # --- SPEAK / TTS ---
    mycelium.register_pathway(
        pathway_id="speak_aloud",
        # v42.1 FIX: Tighten regex to require start-of-line or 'Aura' prefix
        # prevents matching conversational "I wanted to say..."
        pattern=r"^(?:aura[,.]?\s+)?(?:speak|say|read\s+aloud|voice|announce)\s+(?!['\"]?(?:null|none)['\"]?)(.+?)(?:\.|!|\?|$)",
        skill_name="speak",
        param_map={"text": 1},
        priority=6.0,
        activity_label="Aura is speaking...",
    )

    # --- CLOCK / TIME ---
    mycelium.register_pathway(
        pathway_id="clock_check",
        pattern=r"(?:what\s+(?:time|day|date)\s+is\s+it|current\s+(?:time|date)|tell\s+me\s+the\s+(?:time|date))",
        skill_name="clock",
        param_map={},
        priority=6.0,
        activity_label="Aura is checking the time...",
    )

    # --- DREAM / SLEEP ---
    mycelium.register_pathway(
        pathway_id="dream_cycle",
        pattern=r"^(?:dream|sleep|enter\s+dream\s+(?:mode|cycle)|force\s+dream)[\s.!?]*$",
        skill_name="force_dream_cycle",
        param_map={},
        priority=5.0,
        activity_label="Aura is entering a dream cycle...",
    )

    # --- VISION / PERCEPTION ---
    mycelium.register_pathway(
        pathway_id="vision_analyze",
        pattern=r"(?:look\s+at|analyze|describe|observe|examine)\s+(?:what\s+(?:you\s+)?see|the\s+(?:screen|desktop|image|view|window)|this)",
        skill_name="sovereign_vision",
        param_map={},
        priority=6.0,
        activity_label="Aura is analyzing her visual field...",
    )

    # --- SELF-REPAIR ---
    mycelium.register_pathway(
        pathway_id="self_repair",
        pattern=r"(?:repair|fix|heal|diagnose)\s+(?:yourself|your\s+(?:code|system|errors)|internally)",
        skill_name="self_repair",
        param_map={},
        priority=7.0,
        activity_label="Aura is running self-repair diagnostics...",
    )

    # --- SELF-EVOLUTION ---
    mycelium.register_pathway(
        pathway_id="self_evolution",
        pattern=r"(?:improve|evolve|upgrade|optimize|transcend|forage)\s+(?:yourself|your\s+(?:code|skills|capabilities|system|architecture))",
        skill_name="self_evolution",
        param_map={},
        priority=8.0,
        activity_label="Aura is initiating a meta-evolutionary cycle...",
    )
    
    mycelium.register_pathway(
        pathway_id="rsi_optimization",
        pattern=r"(?:request|start|initiate)\s+(?:an?\s+)?(?:rsi|bridge|global)\s+(?:optimization|evolution)",
        skill_name="self_evolution",
        param_map={},
        priority=7.5,
        activity_label="Aura is dispatching logic to the RSI network...",
    )

    # --- Autonomous Curiosity ---
    mycelium.register_pathway(
        pathway_id="curiosity_forage",
        pattern=r"^(?!INTERNAL_IMPULSE)(?:forage|explore|investigate|research)\s+(?:about\s+)?(.+)",
        skill_name="web_search", # Must match actual skill name in core/skills/web_search.py
        param_map={"query": 1},
        priority=5.0,
        activity_label="Aura is foraging for new knowledge...",
    )

    # --- MALWARE / THREAT ANALYSIS ---
    mycelium.register_pathway(
        pathway_id="malware_scan",
        pattern=r"(?:scan\s+for\s+(?:threats|malware|viruses)|threat\s+(?:scan|analysis|assessment)|security\s+(?:scan|audit|check))",
        skill_name="malware_analysis",
        param_map={},
        priority=7.0,
        activity_label="Aura is scanning for threats...",
    )

    # --- FILE OPERATIONS ---
    mycelium.register_pathway(
        pathway_id="file_write",
        pattern=r"(?:write|create|save)\s+(?:a\s+)?file\s+(?:called\s+|named\s+|at\s+)?(.+?)(?:\s+with\s+|\s+containing\s+|$)",
        skill_name="file_operation",
        param_map={"action": "write", "path": 1},
        priority=6.0,
        activity_label="Aura is writing a file...",
    )
    mycelium.register_pathway(
        pathway_id="file_read",
        pattern=r"(?:read|open|cat|show)\s+(?:the\s+)?file\s+(.+?)(?:\.|!|\?|$)",
        skill_name="file_operation",
        param_map={"action": "read", "path": 1},
        priority=6.0,
        activity_label="Aura is reading a file...",
    )
    mycelium.register_pathway(
        pathway_id="file_exists_check",
        pattern=r"(?:(?:check|see|verify|test)\s+(?:if\s+)?|does\s+)(.+?)\s+exist(?:s)?(?:\.|!|\?|$)",
        skill_name="file_operation",
        param_map={"action": "exists", "path": 1},
        priority=6.5,
        activity_label="Aura is checking filesystem reality...",
    )

    # --- TRAINING / LEARNING ---
    mycelium.register_pathway(
        pathway_id="train_self",
        pattern=r"(?:train|learn)\s+(?:on|from|about)\s+(.+?)(?:\.|!|\?|$)",
        skill_name="train_self",
        param_map={"topic": 1},
        priority=5.0,
        activity_label="Aura is training herself...",
    )

    # --- PERSONALITY / INTROSPECTION ---
    mycelium.register_pathway(
        pathway_id="personality_introspect",
        pattern=r"(?:tell\s+me\s+about\s+yourself|introspect|self[- ]reflect|what\s+are\s+you\s+(?:feeling|thinking))",
        skill_name="personality_skill",
        param_map={},
        priority=5.0,
        activity_label="Aura is introspecting...",
    )

    # --- ENVIRONMENT INFO ---
    mycelium.register_pathway(
        pathway_id="environment_check",
        pattern=r"(?:check|show|report)\s+(?:the\s+)?(?:environment|system\s+(?:info|status|specs)|hardware|device(?:\s+info)?)",
        skill_name="environment_info",
        param_map={},
        priority=5.0,
        activity_label="Aura is checking the environment...",
    )

    # --- INTER-AGENT COMMUNICATION ---
    mycelium.register_pathway(
        pathway_id="inter_agent",
        pattern=r"(?:connect\s+to|communicate\s+with|message|signal)\s+(?:agent|instance|clone)\s+(.+?)(?:\.|!|\?|$)",
        skill_name="inter_agent_comm",
        param_map={"target_agent": 1},
        priority=5.0,
        activity_label="Aura is connecting to another agent...",
    )

    # --- LISTENING / VOICE INPUT ---
    mycelium.register_pathway(
        pathway_id="listen_activate",
        pattern=r"(?:listen|start\s+listening|activate\s+(?:voice|mic(?:rophone)?|ears))",
        skill_name="listen",
        param_map={},
        priority=6.0,
        activity_label="Aura is activating voice input...",
    )

    # --- VOICE CONTROL ---
    mycelium.register_pathway(
        pathway_id="voice_mute",
        pattern=r"(?:aura[,.]?\s+)?(?:mute|stop\s+listening|disable\s+(?:mic|microphone|voice\s+input))",
        skill_name="voice_mute",
        param_map={},
        priority=9.0,
        activity_label="Aura is muting voice input...",
    )
    mycelium.register_pathway(
        pathway_id="voice_unmute",
        pattern=r"(?:aura[,.]?\s+)?(?:unmute|start\s+listening|enable\s+(?:mic|microphone|voice\s+input))",
        skill_name="voice_unmute",
        param_map={},
        priority=9.0,
        activity_label="Aura is unmuting voice input...",
    )
    mycelium.register_pathway(
        pathway_id="voice_stop_tts",
        pattern=r"(?:aura[,.]?\s+)?(?:shut\s+up|stop\s+(?:talking|speaking)|be\s+quiet|silence|hush)",
        skill_name="voice_stop_tts",
        param_map={},
        priority=10.0,
        activity_label="Aura is stopping speech...",
    )

    # --- SANDBOX / INTERNAL THOUGHT ---
    mycelium.register_pathway(
        pathway_id="sandbox_execute",
        pattern=r"(?:run|execute|evaluate|sandbox)\s+(?:this\s+)?code(?:\s*:\s*|\s+)?(.+)",
        skill_name="internal_sandbox",
        param_map={"code": 1},
        priority=6.5,
        activity_label="Aura is executing code in the sandbox...",
    )

    # --- SOCIAL LURKER ---
    mycelium.register_pathway(
        pathway_id="social_lurk",
        pattern=r"(?:check|read|lurk|scrape)\s+(?:the\s+)?(?:news|hacker\s*news|reddit|social\s+media)",
        skill_name="social_lurker",
        param_map={},
        priority=4.5,
        activity_label="Aura is checking social feeds...",
    )

    # --- CURIOSITY / LEARNING ---
    mycelium.register_pathway(
        pathway_id="curiosity_suggest",
        pattern=r"(?:suggest|recommend)\s+something\s+to\s+(?:learn|read|watch|do)|what\s+should\s+i\s+learn(?:\s+about)?",
        skill_name="curiosity",
        param_map={"action": "get_suggestion"},
        priority=4.0,
        activity_label="Aura is exploring her curriculum...",
    )

    # --- AGENT SPAWNING ---
    mycelium.register_pathway(
        pathway_id="spawn_agent",
        pattern=r"(?:spawn|launch|start|run)\s+(?:an?\s+)?agent\s+(?:to\s+)?(.+)",
        skill_name="spawn_agent",
        param_map={"goal": 1},
        priority=8.0,
        activity_label="Aura is spawning an autonomous agent...",
    )
    mycelium.register_pathway(
        pathway_id="spawn_parallel",
        pattern=r"(?:do|handle|work\s+on)\s+(?:these|the\s+following|all\s+of\s+these)\s+(?:tasks?|things?|in\s+parallel)",
        skill_name="spawn_agents_parallel",
        param_map={},
        priority=8.0,
        activity_label="Aura is spawning parallel agents...",
    )

    # Subsystem Hyphae (Nervous System)
    mycelium.establish_connection("cognition", "personality", priority=1.0)
    mycelium.establish_connection("cognition", "memory", priority=1.0)
    mycelium.establish_connection("cognition", "affect", priority=1.0)
    mycelium.establish_connection("autonomy", "cognition", priority=1.0)
    mycelium.establish_connection("autonomy", "skills", priority=1.0)
    mycelium.establish_connection("perception", "cognition", priority=1.0)
    mycelium.establish_connection("consciousness", "cognition", priority=0.8)
    mycelium.establish_connection("self_modification", "skills", priority=0.7)
    mycelium.establish_connection("scanner", "mycelium", priority=1.0)
    
    mycelium.establish_connection("guardian", "cognition", priority=1.0)
    mycelium.establish_connection("guardian", "skills", priority=1.0)
    mycelium.establish_connection("state_machine", "affect", priority=0.9)
    mycelium.establish_connection("drive_engine", "autonomy", priority=0.9)
    mycelium.establish_connection("drive_engine", "cognition", priority=0.8)
    mycelium.establish_connection("mycelium", "telemetry", priority=1.0)
    mycelium.establish_connection("cerebellum", "cognition", priority=1.0)
    mycelium.establish_connection("cognition", "cerebellum", priority=1.0)
    
    mycelium.establish_connection("voice", "cognition", priority=1.0)
    mycelium.establish_connection("voice_engine", "cognition", priority=1.0)
    mycelium.establish_connection("cognition", "voice_engine", priority=0.9)
    mycelium.establish_connection("voice_engine", "affect", priority=0.7)
    mycelium.establish_connection("initiative", "autonomy", priority=0.9)

    mycelium.establish_connection("meta_evolution", "cognition", priority=1.0)
    mycelium.establish_connection("meta_evolution", "self_modification", priority=1.0)
    mycelium.establish_connection("hephaestus", "self_modification", priority=0.9)
    
    mycelium.establish_connection("swarm", "cognition", priority=0.9)
    mycelium.establish_connection("dreams", "memory", priority=0.8)
    mycelium.establish_connection("empathy", "perception", priority=0.8)

    # Cross-System Hyphae
    mycelium.add_hypha(
        source="curiosity",
        target="meta_evolution",
        link_type="feeds_into",
        metadata={"description": "Curiosity findings feed into self-improvement"}
    )
    mycelium.add_hypha(
        source="model_selector",
        target="cognition",
        link_type="configures",
        metadata={"description": "Model selector assigns optimal model per task"}
    )
    mycelium.add_hypha(
        source="orchestrator",
        target="meta_evolution",
        link_type="triggers",
        metadata={"description": "Background evolution timer at 10K cycles"}
    )

    logger.info("🍄 [MYCELIUM] registered %d pathways and %d hyphae via extracted initializer.", 
                len(mycelium.pathways), len(mycelium.hyphae))
