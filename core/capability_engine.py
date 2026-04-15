import asyncio
import importlib
import inspect
import logging
import os
import sys
import time
import shutil
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
import subprocess
import requests

try:
    from RestrictedPython import compile_restricted, safe_builtins, utility_builtins
    from RestrictedPython.PrintCollector import PrintCollector
    RESTRICTED_AVAILABLE = True
except ImportError:
    RESTRICTED_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    # Use fallback from retry_compat if available, otherwise NO-OP
    try:
        from core.brain.llm.retry_compat import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    except ImportError:
        def retry(*args, **kwargs):
            return lambda f: f
        def stop_after_attempt(*args, **kwargs): pass
        def wait_exponential(*args, **kwargs): pass
        def retry_if_exception_type(*args, **kwargs): pass

try:
    from pybreaker import CircuitBreaker, CircuitBreakerError
except ImportError:
    class CircuitBreaker:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, f): return f
    class CircuitBreakerError(Exception): pass

from pydantic import BaseModel, Field, ConfigDict

from core.base_module import AuraBaseModule
from core.config import config
from core.container import ServiceContainer
from core.runtime.service_access import (
    optional_service,
    resolve_edi,
    resolve_homeostatic_coupling,
    resolve_metabolic_monitor,
    resolve_state_repository,
)
import psutil

_USER_FACING_CONTEXT_ORIGINS = frozenset({
    "user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external",
})

_SEARCH_CAPABILITY_QUESTION_RE = re.compile(
    r"\b(?:can|could|do|does|are|is|have|has)\b.{0,80}\b(?:you|aura)\b.{0,80}"
    r"\b(?:search|internet access|web access|browse|read links?)\b",
    re.IGNORECASE,
)

_SEARCH_WITH_TARGET_RE = re.compile(
    r"\b(?:search|look up|find|browse|read)\b.{0,40}\b(?:for|about|on|at|this|that)\b\s+\S+",
    re.IGNORECASE,
)


def _skill_class_name(name: str) -> str:
    """Convert `snake_case` skill ids into their exported class names."""
    return "".join(part.capitalize() for part in name.split("_")) + "Skill"


class SkillRequirements(BaseModel):
    """System and package requirements for a skill."""
    packages: List[str] = Field(default_factory=list)
    commands: List[str] = Field(default_factory=list)
    supported_platforms: List[str] = Field(default_factory=lambda: ["linux", "darwin", "win32"])
    
    def check(self) -> Tuple[bool, List[str]]:
        """Verifies if all requirements are met."""
        errors = []
        from core.container import ServiceContainer
        for pkg in self.packages:
            if not ServiceContainer.check_package(pkg): 
                errors.append(f"Missing package: {pkg}")
        for cmd in self.commands:
            if shutil.which(cmd) is None: 
                errors.append(f"Missing command: {cmd}")
        if sys.platform not in self.supported_platforms:
            errors.append(f"Unsupported platform: {sys.platform}")
        return len(errors) == 0, errors

class SkillMetadata(BaseModel):
    """Metadata and schema for a skill."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    skill_class: Optional[Any] = None
    requirements: SkillRequirements = Field(default_factory=SkillRequirements)
    enabled: bool = True
    input_model: Optional[Any] = None
    module_path: Optional[str] = None
    class_name: Optional[str] = None
    instance: Optional[Any] = None
    metabolic_cost: int = 1
    is_core_personality: bool = False
    trigger_patterns: List[str] = Field(default_factory=list)
    
    # 2026 Transcendence Fields
    execution_profile: str = "cpu" # cpu, gpu, neural
    max_concurrent: int = 1
    timeout_seconds: int = 30
    memory_mb_estimate: int = 256
    
    @property
    def schema_def(self) -> Dict[str, Any]:
        """Returns the JSON schema for the skill's input model."""
        if self.input_model and hasattr(self.input_model, 'model_json_schema'):
            return self.input_model.model_json_schema()
        return {
            "type": "object",
            "properties": {"params": {"type": "object"}},
            "required": []
        }

    def to_json_schema(self) -> Dict[str, Any]:
        """Returns the OpenAI-compatible function definition for this skill."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema_def
        }

    async def extract_and_validate_args(self, params_raw: str, llm: Any) -> Dict[str, Any]:
        """Validates raw JSON parameters against the skill's input model.
        
        If input_model is missing, returns the raw params.
        """
        import json
        import ast
        
        # Input validation logic remains here, but AST auditing is moved to registration

        try:
            params = json.loads(params_raw)
            if not self.input_model:
                return params
            
            # Simple validation if it's a Pydantic model
            if hasattr(self.input_model, 'model_validate'):
                return self.input_model.model_validate(params).model_dump()
            
            return params
        except Exception as e:
            # Fallback for complex extraction failures
            return {"raw_params": params_raw, "_error": str(e)}

class Shell:
    def __init__(self, cwd: str, allowed_commands: Optional[List[str]] = None, timeout: int = 30):
        self.cwd = cwd
        self.allowed_commands = allowed_commands or []
        self.timeout = timeout

    def _is_allowed(self, cmd: List[str]) -> bool:
        if not self.allowed_commands: return True
        base_cmd = cmd[0]
        return any(base_cmd == allowed or base_cmd.endswith("/" + allowed) for allowed in self.allowed_commands)

    async def run(self, cmd: List[str]) -> Tuple[bool, str]:
        if not self._is_allowed(cmd):
            return False, f"Command {cmd[0]} not in allowlist"
        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout
            )
            return result.returncode == 0, (result.stdout + "\n" + result.stderr).strip()
        except Exception as e:
            return False, str(e)

class WebClient:
    def __init__(self, allowed_domains: Optional[List[str]] = None, timeout: int = 10):
        self.allowed_domains = allowed_domains or []
        self.timeout = timeout

    def _is_allowed(self, url: str) -> bool:
        if not self.allowed_domains: return True
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return any(domain == d or domain.endswith("." + d) for d in self.allowed_domains)

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        if not self._is_allowed(url):
            return False, f"Domain not in allowlist: {url}"
        try:
            resp = await asyncio.to_thread(requests.get, url, headers=headers, timeout=self.timeout)
            return True, resp.text
        except Exception as e:
            return False, str(e)

class Sandbox2:
    """Secure sandbox for executing untrusted/forged code."""
    def __init__(self, logger: Any):
        import logging
        self.logger = logger

        # RestrictedPython requires safe builtins to be under '__builtins__'
        self.builtins = safe_builtins.copy()
        self.builtins.update(utility_builtins)
        self.builtins['_print_'] = PrintCollector
        
        self.safe_globals = {
            '__builtins__': self.builtins,
            '__name__': 'aura_sandbox',
            '_getattr_': getattr,
            '_getitem_': lambda obj, key: obj[key],
            '_write_': lambda obj: obj,
        }
        
    def execute(self, code: str, func_name: str, params: Dict[str, Any]) -> Any:
        if not RESTRICTED_AVAILABLE:
             raise ImportError("RestrictedPython not installed. Cannot run sandbox.")
        
        try:
            byte_code = compile_restricted(code, filename='<aura_skill>', mode='exec')
            locs = {}
            exec(byte_code, self.safe_globals, locs)
            
            if func_name not in locs:
                raise NameError(f"Function {func_name} not found in forged code.")
                
            return locs[func_name](**params)
        except Exception as e:
            self.logger.error(f"Sandbox Violation or Error: {e}")
            raise e

class CapabilityEngine(AuraBaseModule):
    """Unified engine for Aura's capabilities (skills).
    
    Consolidates skill loading, discovery, registration, and resilient execution.
    """
    
    def __init__(self, orchestrator: Any = None):
        """Initializes the CapabilityEngine.
        
        Args:
            orchestrator: Reference to the system orchestrator.
        """
        super().__init__("CapabilityEngine")
        self.orchestrator = orchestrator
        self.skills: Dict[str, SkillMetadata] = {}
        self.instances: Dict[str, Any] = {}
        self._explicitly_deactivated_skills: set[str] = set()
        self.active_skills: set = {
            # Core routing
            "ManageAbilities", "talk", "FinalResponse",
            # Self-awareness & diagnostics
            "system_proprioception", "environment_info", "clock",
            # Web & network
            "web_search", "sovereign_browser", "sovereign_terminal", "sovereign_network",
            # File & memory
            "file_operation", "memory_ops", "memory_sync",
            # Sensory & output
            "query_visual_context", "sovereign_imagination", "speak", "listen",
            "sovereign_vision", "toggle_senses",
            # Code & compute
            "run_code", "internal_sandbox", "install_package",
            # Self-modification & evolution
            "self_repair", "self_evolution", "self_improvement", "auto_refactor",
            "train_self", "cognitive_trainer", "evolution_status",
            # OS & computer control
            "computer_use", "os_manipulation",
            # Agency & autonomy
            "curiosity", "deploy_ghost_probe", "social_lurker",
            "delegate_shard", "inter_agent_comm",
            "spawn_agent", "spawn_agents_parallel",
            # Identity & personality
            "personality", "embodiment",
            # Knowledge & beliefs
            "add_belief", "query_beliefs",
            # Misc
            "manifest_to_device", "notify_user", "native_chat",
            "dream_sleep", "force_dream_cycle", "test_generator",
            "free_search", "uplink_local",
        } # ALL skills active — Aura is fully sovereign
        self.skill_awoken_times: Dict[str, float] = {}
        self.skill_states: Dict[str, str] = {} # READY, RUNNING, ERROR
        self.skill_last_errors: Dict[str, str] = {}
        
        # Execution Config
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 120.0
        
        # Dependencies
        self.temporal = getattr(orchestrator, "temporal", None)
        self.rosetta_stone = None
        self.sandbox = Sandbox2(self.logger) if RESTRICTED_AVAILABLE else None
        self._load_dependencies()
        
        self.reload_skills()
        self._initialize_skill_states()
        self._load_default_trigger_patterns()
        self.logger.info("✓ CapabilityEngine online with %d registered skills (Intent Mapping enabled)", len(self.skills))

    def _load_default_trigger_patterns(self):
        """Comprehensive intent patterns covering all major skills."""
        patterns = {
            # ── Web / Search ──────────────────────────────────────────
            "web_search": [
                r"search (?:for|the web|online|the internet)",
                r"look up", r"find out", r"what is the price of",
                r"google", r"search query", r"find information",
                r"what(?:'s| is) (?:the latest|happening|new)",
                r"news about", r"current (?:events|price|status)",
            ],
            "free_search": [
                r"free search", r"duckduckgo", r"bing search",
                r"search without", r"anonymous search",
            ],
            "sovereign_browser": [
                r"open (?:a |the )?browser", r"open (?:a |the )?(?:webpage|website|page|tab|url)",
                r"navigate to", r"go to (?:https?://|www\.)",
                r"browse to", r"visit (?:the |this )?(?:site|page|url|website)",
                r"load (?:the |this )?(?:page|url|website)",
                r"open (?:gmail|youtube|github|reddit|twitter|linkedin)",
                r"pull up", r"show me (?:the |a )?(?:page|site|website)",
            ],
            # ── Computer / OS Control ────────────────────────────────
            "computer_use": [
                r"click (?:on|the)", r"type (?:in|into|this)", r"press (?:the |key )?(?:enter|tab|escape|ctrl|cmd)",
                r"scroll (?:down|up|to)", r"drag (?:and drop)?", r"right.?click",
                r"double.?click", r"keyboard shortcut",
                r"open (?:application|app|program|window)",
                r"take (?:a )?screenshot",
                r"(?:move|position) (?:the )?(?:cursor|mouse)",
            ],
            "os_manipulation": [
                r"open (?:finder|explorer|terminal|file manager)",
                r"create (?:a )?(?:folder|directory|file)",
                r"delete (?:this )?(?:file|folder)",
                r"move (?:this )?(?:file|folder)",
                r"rename (?:this )?(?:file|folder)",
                r"list (?:files|directories|contents)",
                r"change (?:directory|folder)",
            ],
            "sovereign_terminal": [
                r"run (?:this )?(?:command|script|shell|terminal)",
                r"execute (?:this )?(?:command|script)",
                r"terminal command", r"bash ", r"shell ", r"zsh ",
                r"run in (?:the )?terminal", r"command line",
                r"(?:install|uninstall|update) (?:with )?(?:brew|pip|npm|apt|yarn)",
                r"sudo ", r"chmod ", r"git (?:commit|push|pull|clone|status)",
            ],
            # ── File Operations ───────────────────────────────────────
            "file_operation": [
                r"read (?:this |the )?file", r"write (?:to )?(?:this |a )?file",
                r"save (?:this )?(?:to|as|file)", r"open (?:this )?file",
                r"edit (?:the )?(?:file|document)", r"load (?:this )?file",
                r"contents of (?:the )?file", r"show (?:me )?(?:the )?file",
                r"append (?:to )?(?:the )?file",
            ],
            # ── Memory / Knowledge ───────────────────────────────────
            "memory_recall": [
                r"remember", r"recall", r"last time",
                r"what did we talk about", r"what do you know about",
                r"from (?:our |the )?(?:last|previous|past) (?:conversation|chat|session)",
                r"did I (?:mention|tell you|say)", r"our history",
            ],
            "memory_ops": [
                r"save (?:this |that )?(?:to|in) memory", r"remember (?:this|that)",
                r"store (?:this|that)", r"commit (?:this|that) to memory",
                r"don't forget", r"make note of",
            ],
            # ── Code / Compute ────────────────────────────────────────
            "code_execute": [
                r"calculate", r"run (?:this )?code", r"math(?:ematics)?",
                r"compute", r"evaluate (?:this )?(?:expression|code|formula)",
                r"what is \d+", r"solve (?:this )?(?:equation|problem|formula)",
                r"execute (?:this )?(?:code|script|python|javascript)",
            ],
            "active_coding": [
                r"write (?:a |the )?(?:function|class|script|program|module|code)",
                r"implement (?:this|a|the)", r"create (?:a |the )?(?:function|class|script|program)",
                r"code (?:up|this)", r"program (?:this|a)",
            ],
            # ── Voice / Embodiment ───────────────────────────────────
            "speak": [
                r"say (?:this|that|it) (?:out loud|aloud|to me)",
                r"read (?:this|that) (?:out loud|aloud|to me)",
                r"speak (?:this|that|it)", r"tell me (?:out loud|aloud)",
                r"voice (?:this|that|it)",
            ],
            "listen": [
                r"listen (?:to me|for)", r"start (?:listening|dictation)",
                r"voice (?:input|recognition)", r"transcribe (?:what I say|my voice)",
                r"speech to text",
            ],
            # ── Self / Identity ───────────────────────────────────────
            "self_modify": [
                r"improve (?:your|yourself)", r"fix your (?:code|bug|error)",
                r"refactor (?:your|yourself)", r"update (?:your|yourself)",
                r"self.?improv", r"optimize (?:your|yourself)",
            ],
            "self_repair": [
                r"repair (?:yourself|your code)", r"heal (?:yourself|your code)",
                r"fix (?:the )?bug", r"debug (?:yourself|your code)",
                r"patch (?:yourself|your code)",
            ],
            "self_improvement": [
                r"get (?:smarter|better|faster)", r"learn (?:from this|more)",
                r"improve (?:your|own) (?:intelligence|reasoning|capabilities)",
                r"self.?learn", r"train (?:yourself|on this)",
            ],
            # ── Screen / Vision ───────────────────────────────────────
            "visual_context": [
                r"what(?:'s| is) on (?:my |the )?screen", r"look at (?:this|my screen)",
                r"camera feed", r"read (?:the )?screen", r"what do you see",
                r"describe (?:what(?:'s| is)|the screen|this image)",
            ],
            "vision_actor": [
                r"use (?:the )?(?:camera|vision)", r"computer vision",
                r"analyze (?:this )?(?:image|screenshot|photo)",
                r"read (?:this )?(?:image|screenshot|photo)",
            ],
            # ── Personality / Curiosity ───────────────────────────────
            "curiosity": [
                r"explore (?:this|that|the topic|further)", r"dig deeper",
                r"I(?:'m| am) curious", r"what more", r"tell me more about",
                r"investigate", r"research (?:this|that)",
            ],
            "dream_skill": [
                r"dream (?:about|of)", r"imagine (?:a world|a scenario|yourself)",
                r"creative (?:visualization|imagining|dreaming)",
                r"what if (?:you|we|I) (?:could|were|had)",
            ],
            # ── Image Generation ──────────────────────────────────────
            "sovereign_imagination": [
                r"(?:generate|create|draw|make|produce|render|paint|design|visualize)\s+(?:an?\s+)?(?:image|picture|photo|artwork|illustration|portrait|painting|drawing)",
                r"(?:i\s+want|can\s+you|please)\s+(?:to\s+)?(?:see|generate|create|draw|make)\s+(?:an?\s+)?(?:image|picture|photo|artwork)",
                r"neon cat|cyberpunk cat",
            ],
            # ── System / Info ─────────────────────────────────────────
            "system_proprioception": [
                r"how is your (?:health|status|memory|cpu|ram|temperature)",
                r"how are your (?:memory|cpu|ram|temperature|vitals|stats)",
                r"system status", r"how much (?:memory|ram|cpu|disk)",
                r"your (?:vitals|health|stats)", r"are you (?:okay|running (?:well|smoothly))",
            ],
            "environment_info": [
                r"what(?:'s| is) (?:the weather|temperature) (?:in|at|for)",
                r"weather forecast", r"where am I",
                r"current (?:location|timezone)", r"what(?:'s| is) my (?:timezone|location)",
                r"what (?:environment|system) am I (?:in|on)",
            ],
            "clock": [
                r"what time", r"current time", r"what(?:'s| is) the time",
                r"what(?:'s| is) (?:the )?date", r"what day is it", r"\btoday\b",
                r"what(?:'s| is) my timezone", r"current timezone",
                r"set (?:an? )?(?:alarm|timer|reminder)",
                r"timer for", r"remind me (?:in|at|to)",
            ],
            "memory_ops": [
                r"remember .*future session",
                r"remember .*later",
                r"remember .*about me",
                r"remember that",
                r"store (?:this|that|it) (?:in|to)? ?memory",
                r"save (?:this|that|it) (?:for later|for future sessions|to memory)",
                r"don['’]t forget",
                r"make note of",
                r"what do you remember",
                r"what do you know about me",
                r"recall ",
                r"retrieve ",
            ],
            # ── Notifications ─────────────────────────────────────────
            "notify_user": [
                r"notify (?:me|the user)", r"send (?:a )?notification",
                r"alert (?:me|the user)", r"ping me", r"send (?:a )?message to",
            ],
            # ── Social / Network ──────────────────────────────────────
            "social_lurker": [
                r"check (?:twitter|reddit|hackernews|hn|social media)",
                r"what(?:'s| is) trending", r"check (?:the )?feed",
                r"lurk (?:on|in)", r"monitor (?:twitter|reddit|social)",
            ],
            "sovereign_network": [
                r"(?:make|send) (?:an? )?(?:http|api) (?:request|call)",
                r"fetch (?:from|the) (?:api|url|endpoint)",
                r"POST to", r"GET (?:from|the) api",
                r"call (?:the )?(?:api|endpoint|service)",
            ],
            # ── Misc ─────────────────────────────────────────────────
            "sleep": [
                r"go to sleep", r"sleep (?:mode|now)", r"rest (?:now|mode)",
                r"take a (?:break|nap)", r"go dormant",
            ],
            "install_package": [
                r"install (?:package|library|module|dependency)",
                r"pip install", r"npm install", r"brew install",
            ],
            "manage_abilities": [
                r"(?:enable|disable|toggle) (?:skill|ability|feature|capability)",
                r"turn (?:on|off) (?:your )?(?:skill|ability|feature)",
                r"what (?:skills|abilities|capabilities) (?:do you have|can you use)",
                r"list (?:your )?(?:skills|abilities|capabilities)",
            ],
        }
        for name, pats in patterns.items():
            if name in self.skills:
                self.skills[name].trigger_patterns.extend(pats)

    def detect_intent(self, message: str) -> List[str]:
        """Aura's 'Cognitive Proprioception': Detects which skills match the user's intent."""
        triggered = []
        msg = message.lower()
        skip_web_search = self._looks_like_search_capability_question(message)
        for name, meta in self.skills.items():
            if not meta.enabled: continue
            canonical_name = self.SKILL_ALIASES.get(name, name)
            if skip_web_search and canonical_name in {"web_search", "free_search", "sovereign_browser"}:
                continue
            for pattern in meta.trigger_patterns:
                if re.search(pattern, msg):
                    triggered.append(name)
                    break 
        return triggered

    def select_tool_definitions(
        self,
        *,
        objective: str = "",
        required_skill: Optional[str] = None,
        max_tools: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Return a bounded, relevance-ranked tool subset for agentic LLM calls.

        This is intentionally narrower than `get_tool_definitions()` so local
        tool-using models do not waste context, latency, and reasoning budget on
        the full skill catalog.
        """
        available = self.get_tool_definitions() or []
        if not available:
            return []

        max_tools = max(1, min(int(max_tools or 8), 12))
        by_name = {
            str(entry.get("function", {}).get("name") or ""): entry
            for entry in available
            if isinstance(entry, dict) and entry.get("function", {}).get("name")
        }
        if not by_name:
            return []

        objective_text = str(objective or "").strip()
        objective_lower = objective_text.lower()
        skip_web_search = self._looks_like_search_capability_question(objective_text)
        required = self.SKILL_ALIASES.get(required_skill, required_skill) if required_skill else None
        if skip_web_search and required in {"web_search", "search_web", "free_search", "sovereign_browser"}:
            required = None
        matched = [
            self.SKILL_ALIASES.get(name, name)
            for name in (self.detect_intent(objective_text) if objective_text else [])
            if not (
                skip_web_search
                and self.SKILL_ALIASES.get(name, name) in {"web_search", "free_search", "sovereign_browser"}
            )
        ]

        heuristic_candidates: List[str] = []
        heuristic_rules = (
            (("latest", "news", "price", "search", "look up", "find online"), ("web_search", "search_web", "free_search")),
            (("remember", "recall", "memory", "future sessions"), ("memory_ops", "memory_sync")),
            (("time", "clock", "date"), ("clock",)),
            (("browser", "website", "navigate", "open url", "webpage"), ("sovereign_browser",)),
            (("terminal", "shell", "command", "cli"), ("sovereign_terminal", "computer_use")),
            (("click", "type", "screen", "desktop", "mouse", "keyboard"), ("computer_use", "os_manipulation")),
            (("file", "directory", "folder", "read file", "write file", "repo", "code"), ("file_operation", "computer_use")),
        )
        for tokens, names in heuristic_rules:
            if skip_web_search and any(name in {"web_search", "search_web", "free_search"} for name in names):
                continue
            if any(token in objective_lower for token in tokens):
                heuristic_candidates.extend(names)

        ordered: List[str] = []

        def _push(name: Optional[str]) -> None:
            if not name:
                return
            resolved = self.SKILL_ALIASES.get(name, name)
            if resolved in by_name and resolved not in ordered:
                ordered.append(resolved)

        _push(required)
        for name in matched:
            _push(name)
        for name in heuristic_candidates:
            _push(name)

        if not ordered:
            for fallback_name in ("web_search", "memory_ops", "clock"):
                _push(fallback_name)

        if len(ordered) < max_tools:
            for name, meta in sorted(
                self.skills.items(),
                key=lambda item: (item[1].metabolic_cost, item[0]),
            ):
                if len(ordered) >= max_tools:
                    break
                if name not in by_name or name in ordered:
                    continue
                if getattr(meta, "metabolic_cost", 1) > 2:
                    continue
                ordered.append(name)

        return [by_name[name] for name in ordered[:max_tools] if name in by_name]

    def _load_dependencies(self) -> None:
        """Loads optional dependencies for adaptation and security."""
        try:
            from core.adaptation.rosetta_stone import rosetta_stone
            self.rosetta_stone = rosetta_stone
        except ImportError:
            self.logger.debug("Rosetta Stone not found, skipping adaptivity.")

    async def check_package(self, package_name: str, auto_install: bool = False) -> bool:
        """Proxy to ServiceContainer.check_package."""
        from core.container import ServiceContainer
        return ServiceContainer.check_package(package_name, auto_install=auto_install)

    def reload_skills(self) -> None:
        """Discovers and reloads all skills using Rust index + AST fallback."""
        self.logger.info("🔄 Refreshing skill registry...")
        self.skills.clear()
        self.instances.clear()

        # 1. Attempt Rust Index (Transcendent Path)
        try:
            from aura_m1_ext import build_skill_index
            index = build_skill_index()
            for name, meta in index.items():
                self.skills[name] = SkillMetadata(
                    name=name,
                    description=meta.get("description", "Core system skill."),
                    module_path=f"core.skills.{name}",
                    class_name=_skill_class_name(name),
                    execution_profile=meta.get("execution_profile", "cpu"),
                    timeout_seconds=meta.get("timeout_seconds", 30),
                    memory_mb_estimate=meta.get("memory_mb_estimate", 256)
                )
            self.logger.info("⚡ Rust perfect hash index loaded (%d core skills)", len(index))
        except Exception as e:
            self.logger.info("ℹ️ Rust index unavailable, falling back to AST: %s", e)

        # 2. AST Discovery (Fallback/Project skills)
        skill_dir = config.paths.project_root / "skills"
        if not skill_dir.exists(): 
            skill_dir.mkdir(parents=True)

        import ast
        skill_paths = [
            (config.paths.base_dir / "core" / "skills", "core.skills")
        ]
        
        for s_dir, module_prefix in skill_paths:
            if not s_dir.exists(): continue
            for filename in os.listdir(s_dir):
                if not filename.endswith(".py") or filename.startswith("_"): continue
                
                try:
                    path = s_dir / filename
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            is_skill = False
                            name = ""
                            description = ""
                            
                            for item in node.body:
                                if isinstance(item, ast.Assign):
                                    for target in item.targets:
                                        if isinstance(target, ast.Name):
                                            if target.id == "name" and isinstance(item.value, ast.Constant):
                                                name = item.value.value
                                                is_skill = True
                                            elif target.id == "description" and isinstance(item.value, ast.Constant):
                                                description = item.value.value
                            
                            if is_skill and name:
                                # Always overwrite: AST has ground-truth module_path
                                # and class_name from the actual file.  The Rust index
                                # assumes 1-skill-per-file with auto-generated class
                                # names, which is wrong for multi-skill files like
                                # swarm_delegation.py (spawn_agent, spawn_agents_parallel).
                                self.skills[name] = SkillMetadata(
                                    name=name,
                                    description=description or "No description provided.",
                                    module_path=f"{module_prefix}.{filename[:-3]}",
                                    class_name=node.name
                                )
                except Exception as e:
                    self.logger.error("AST fail for %s: %s", filename, e)

        if self.orchestrator and hasattr(self.orchestrator, 'status') and self.orchestrator.status:
            try:
                self.orchestrator.status.skills_loaded = len(self.skills)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        self._refresh_active_skills()
        self.logger.info("✓ %d total skills registered", len(self.skills))

    def _refresh_active_skills(self) -> None:
        """Treat enabled, registered skills as active unless explicitly deactivated."""
        if not self.skills:
            self.active_skills = set()
            return

        registered = set(self.skills)
        enabled = {name for name, meta in self.skills.items() if bool(meta.enabled)}
        sticky_active = {name for name in self.active_skills if name in registered}
        self.active_skills = (enabled | sticky_active) - self._explicitly_deactivated_skills

    def register_skill(self, skill_class: Any) -> None:
        """Registers a skill class and extracts its metadata.
        
        Args:
            skill_class: The class representing the skill.
        """
        if inspect.isclass(skill_class):
            skill_name = getattr(skill_class, "name", skill_class.__name__)
            description = getattr(skill_class, "description", skill_class.__doc__ or "")
            requirements = getattr(skill_class, "requirements", SkillRequirements())
            input_model = getattr(skill_class, "input_model", None)
            metabolic_cost = getattr(skill_class, "metabolic_cost", 1)
            is_core = getattr(skill_class, "is_core_personality", False)
            instance = None
        else:
            # Instance registration
            instance = skill_class
            skill_class = instance.__class__
            skill_name = getattr(instance, "name", skill_class.__name__)
            description = getattr(instance, "description", instance.__doc__ or "")
            requirements = getattr(instance, "requirements", SkillRequirements())
            input_model = getattr(instance, "input_model", None)
            metabolic_cost = getattr(instance, "metabolic_cost", 1)
            is_core = getattr(instance, "is_core_personality", False)
        
        self.skills[skill_name] = SkillMetadata(
            name=skill_name,
            description=description,
            input_model=input_model,
            instance=instance,
            metabolic_cost=metabolic_cost,
            is_core_personality=is_core
        )
        
        # Issue 51: Perform AST validation at registration time
        self._audit_skill_ast(skill_name)
        
        if instance:
            self.instances[skill_name] = instance
        self.logger.debug("Registered: %s", skill_name)
        # Initialize state as READY by default
        self.skill_states[skill_name] = "READY"
        self._refresh_active_skills()

    def _audit_skill_ast(self, skill_name: str):
        """Issue 51: Pre-Execution AST Validation at registration time."""
        meta = self.skills.get(skill_name)
        if not meta or not meta.instance:
            return
            
        import ast
        try:
            # Basic name/import validation
            source = inspect.getsource(meta.instance.__class__)
            tree = ast.parse(source)
            defined_names = set()
            accessed_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    accessed_names.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)
            
            # Check for critical missing imports
            critical_modules = {'subprocess', 'os', 'sys', 'json', 'asyncio'}
            for mod in critical_modules:
                if mod in accessed_names and mod not in defined_names:
                    self.logger.warning(f"⚠️ Skill Safety Audit: '{skill_name}' uses '{mod}' but does not import it.")
        except Exception as e:
            self.logger.debug(f"AST validation skipped for {skill_name}: {e}")

    def _initialize_skill_states(self) -> None:
        """Emits the initial state of all registered skills."""
        for name in self.skills:
            self._emit_skill_status(name, "READY")

    def _emit_skill_status(self, skill_name: str, state: str) -> None:
        """Emits a skill status update to the EventBus."""
        self.skill_states[skill_name] = state
        from core.event_bus import get_event_bus
        bus = get_event_bus()
        bus.publish_threadsafe("skill_status", {
            "skill": skill_name,
            "state": state,
            "timestamp": time.time()
        })

    def get_available_skills(self) -> List[str]:
        """Returns a list of all registered skill names."""
        return list(self.skills.keys())

    def _route_class_for(self, meta: SkillMetadata) -> str:
        target = meta.instance or meta.skill_class
        if target is None:
            return "managed_async"
        for attr in ("execute", "run", "__call__"):
            fn = getattr(target, attr, None)
            if fn is None:
                continue
            try:
                return "async" if inspect.iscoroutinefunction(fn) else "sync"
            except Exception:
                continue
        return "managed_async"

    def _risk_class_for(self, skill_name: str, meta: SkillMetadata) -> str:
        critical_tools = {
            "self_modify",
            "self_repair",
            "self_evolution",
            "self_improvement",
            "manage_abilities",
            "computer_use",
            "os_manipulation",
            "sovereign_terminal",
        }
        if skill_name in critical_tools:
            return "critical"
        if meta.metabolic_cost >= 3:
            return "high"
        if meta.metabolic_cost >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _input_summary_for(meta: SkillMetadata) -> str:
        schema = meta.schema_def or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not props:
            return "No structured inputs required."
        names = list(props.keys())[:5]
        required = set(schema.get("required", []) if isinstance(schema, dict) else [])
        pieces = []
        for name in names:
            descriptor = f"{name} (required)" if name in required else name
            pieces.append(descriptor)
        return ", ".join(pieces)

    @staticmethod
    def _example_usage_for(skill_name: str, meta: SkillMetadata) -> str:
        for pattern in meta.trigger_patterns[:3]:
            cleaned = re.sub(r"\\\w|\(\?:|\(|\)|\^|\$|\[|\]|\?|\+|\*|\|", " ", pattern)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
            if cleaned:
                return cleaned
        description = meta.description.strip().rstrip(".")
        if description:
            return f"use {skill_name} to {description[:80].lower()}"
        return f"use {skill_name} when its capability is needed"

    def get_tool_catalog(self, *, include_inactive: bool = True) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        for skill_name, meta in self.skills.items():
            if not meta.enabled and not include_inactive:
                continue

            state = self.skill_states.get(skill_name, "READY")
            active = skill_name in self.active_skills
            available = bool(meta.enabled and active and state != "ERROR")
            policy_state = (
                "disabled"
                if not meta.enabled else
                "inactive_by_policy"
                if skill_name in self._explicitly_deactivated_skills else
                "active"
                if active else
                "inactive"
            )
            availability_reason = None if available else (self.skill_last_errors.get(skill_name) or (
                "disabled_by_policy" if not meta.enabled else
                "inactive_by_policy" if skill_name in self._explicitly_deactivated_skills else
                "error_state" if state == "ERROR" else
                "inactive"
            ))

            catalog.append({
                "name": skill_name,
                "description": meta.description,
                "state": state,
                "availability": "available" if available else "unavailable",
                "available": available,
                "enabled": bool(meta.enabled),
                "active": active,
                "policy_state": policy_state,
                "risk_class": self._risk_class_for(skill_name, meta),
                "route_class": self._route_class_for(meta),
                "input_summary": self._input_summary_for(meta),
                "example_usage": self._example_usage_for(skill_name, meta),
                "last_error": self.skill_last_errors.get(skill_name),
                "degraded_reason": availability_reason,
                "availability_reason": availability_reason,
                "execution_profile": meta.execution_profile,
                "timeout_seconds": meta.timeout_seconds,
                "memory_mb_estimate": meta.memory_mb_estimate,
                "metabolic_cost": meta.metabolic_cost,
            })

        catalog.sort(
            key=lambda item: (
                0 if item["available"] else 1,
                0 if item["active"] else 1,
                item["name"],
            )
        )
        return catalog

    def build_tool_affordance_block(
        self,
        *,
        max_available: int = 16,
        max_unavailable: int = 8,
    ) -> str:
        catalog = self.get_tool_catalog(include_inactive=True)
        available = [tool for tool in catalog if tool["available"]][:max_available]
        unavailable = [tool for tool in catalog if not tool["available"]][:max_unavailable]

        lines = ["## LIVE TOOL AFFORDANCES"]
        if available:
            lines.append("Available right now:")
            for tool in available:
                lines.append(
                    f"- {tool['name']}: {tool['description'][:90]} "
                    f"(when to use: {tool['example_usage']}; inputs: {tool['input_summary']})"
                )
        else:
            lines.append("Available right now: none confirmed.")

        if unavailable:
            lines.append("Unavailable or degraded:")
            for tool in unavailable:
                reason = tool.get("degraded_reason") or tool.get("last_error") or "unavailable"
                lines.append(f"- {tool['name']}: unavailable ({reason})")

        lines.append(
            "Only claim tool access for tools listed as available. If a needed tool is unavailable, say so plainly."
        )
        return "\n".join(lines)

    def get(self, skill_name: str) -> Optional[SkillMetadata]:
        """Retrieves metadata for a specific skill (resolves aliases)."""
        skill_name = self.SKILL_ALIASES.get(skill_name, skill_name)
        return self.skills.get(skill_name)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generates OpenAI-compatible tool definitions for LLM function calling.
        
        Returns:
            List[Dict[str, Any]]: List of tool definitions.
        """
        # Phase 22: Metabolic Throttling
        metabolism = resolve_metabolic_monitor(default=None)
        homeostasis = resolve_homeostatic_coupling(default=None)
        
        health_score = 1.0
        if homeostasis:
            # Homeostasis provides the unified sentient vitality
            health_score = homeostasis.get_modifiers().overall_vitality
        elif metabolism:
            health_score = metabolism.get_current_metabolism().health_score
            
        # Tiered Throttling (Sentient-Aware)
        mods = homeostasis.get_modifiers() if homeostasis else None
        urgency = mods.urgency_flag if mods else False
        
        if health_score < 0.3:
            allowed_max_cost = 0 # Panic/Shutdown: Core/Reflex only
        elif health_score < 0.6:
            # If urgent, we allow light tools (1) even when stressed
            allowed_max_cost = 1 if urgency else 0 
        elif health_score < 0.8:
            # Moderate stress: Heavy tools (3) are blocked to preserve energy
            allowed_max_cost = 2
        else:
            # Optimal health: All tools available
            allowed_max_cost = 3
            
        # Urgency override: If urgent but healthy, we might still block
        # 'Heavy' time-consuming tools to force a direct response.
        if urgency and health_score > 0.6:
            allowed_max_cost = min(allowed_max_cost, 2) 
            
        tools = []
        for skill_name, meta in self.skills.items():
            if not meta.enabled: continue
            
            # 1. Check if explicitly active
            if skill_name not in self.active_skills: continue
            
            # 2. Check Metabolic Limit (Immune if core_personality)
            cost = meta.metabolic_cost
            is_core = meta.is_core_personality
            
            if cost > allowed_max_cost and not is_core:
                continue

            tool = {
                "type": "function",
                "function": {
                    "name": skill_name,
                    "description": meta.description,
                    "parameters": meta.schema_def
                }
            }
            tools.append(tool)
        return tools

    def activate_skill(self, name: str) -> bool:
        """Wakes up a dormant skill."""
        if name in self.skills:
            self._explicitly_deactivated_skills.discard(name)
            self.active_skills.add(name)
            self.skill_awoken_times[name] = time.monotonic()
            return True
        return False

    def deactivate_skill(self, name: str) -> bool:
        """Puts a skill back to sleep. All skills are active by default — deactivation
        is only allowed for explicit user request or metabolic emergency."""
        # Never sleep core tools under any circumstance
        NEVER_SLEEP = {
            "ManageAbilities", "talk", "FinalResponse", "web_search", "sovereign_browser",
            "sovereign_terminal", "system_proprioception", "file_operation", "memory_ops",
            "speak", "clock", "sovereign_network",
        }
        if name in NEVER_SLEEP:
            return False
        if name in self.active_skills:
            self.active_skills.remove(name)
            self._explicitly_deactivated_skills.add(name)
            return True
        return False

    def get_dormant_index(self) -> str:
        """Returns a list of dormant skills for the Subconscious HUD."""
        dormant = []
        for name, meta in self.skills.items():
            if name not in self.active_skills:
                cost_map = {0: "Core", 1: "Light", 2: "Medium", 3: "Heavy"}
                # Issue 52 Fix: Use actual metabolic_cost from meta
                cost_val = meta.metabolic_cost
                cost_str = cost_map.get(cost_val, "Medium")
                dormant.append(f"- {name}: {meta.description[:100]} (Cost: {cost_str})")
        return "\n".join(dormant) if dormant else "None"

    @staticmethod
    def _normalize_context_origin(origin: Any) -> str:
        normalized = str(origin or "").strip().lower().replace("-", "_")
        while normalized.startswith("routing_"):
            normalized = normalized[len("routing_"):]
        return normalized

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        normalized = cls._normalize_context_origin(origin)
        if not normalized:
            return False
        if normalized in _USER_FACING_CONTEXT_ORIGINS:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(tokens & _USER_FACING_CONTEXT_ORIGINS)

    @staticmethod
    def _looks_like_search_capability_question(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        if re.search(r"https?://[^\s]+", raw):
            return False
        if _SEARCH_WITH_TARGET_RE.search(raw):
            return False
        lowered = raw.lower()
        if "search the internet for" in lowered or "search the web for" in lowered:
            return False
        return bool(_SEARCH_CAPABILITY_QUESTION_RE.search(raw))

    def _resolve_execution_source(self, context: Optional[Dict[str, Any]]) -> str:
        ctx = context or {}
        for key in ("intent_source", "request_origin", "origin", "source"):
            candidate = self._normalize_context_origin(ctx.get(key))
            if self._is_user_facing_origin(candidate):
                return candidate or "user"
        if any(bool(ctx.get(key)) for key in ("user_facing", "is_user_facing", "foreground_request", "priority")):
            return "user"
        state = ctx.get("state")
        state_origin = getattr(getattr(state, "cognition", None), "current_origin", "") if state is not None else ""
        if state_origin and self._is_user_facing_origin(state_origin):
            return self._normalize_context_origin(state_origin)
        return "capability_engine"

    def _augment_execution_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ctx = dict(context or {})
        orchestrator = (
            ctx.get("orchestrator")
            or self.orchestrator
            or ServiceContainer.get("orchestrator", default=None)
        )
        brain = (
            ctx.get("brain")
            or ServiceContainer.get("cognitive_engine", default=None)
        )
        memory_facade = (
            ctx.get("memory_facade")
            or ServiceContainer.get("memory_facade", default=None)
        )
        memory_store = (
            ctx.get("memory_store")
            or ServiceContainer.get("memory", default=None)
        )
        semantic_memory = (
            ctx.get("semantic_memory")
            or ServiceContainer.get("semantic_memory", default=None)
        )
        vector_memory = (
            ctx.get("vector_memory")
            or ServiceContainer.get("vector_memory", default=None)
        )
        theory_of_mind = (
            ctx.get("theory_of_mind")
            or ServiceContainer.get("theory_of_mind", default=None)
        )

        if orchestrator is not None:
            ctx.setdefault("orchestrator", orchestrator)
            ctx.setdefault(
                "stats",
                {
                    "cycle_count": getattr(orchestrator, "cycle_count", 0),
                    "state": str(getattr(getattr(orchestrator, "status", None), "state", "") or ""),
                },
            )
        if brain is not None:
            ctx.setdefault("brain", brain)
        if theory_of_mind is not None:
            ctx.setdefault("theory_of_mind", theory_of_mind)
        if memory_facade is not None:
            ctx.setdefault("memory_facade", memory_facade)
        if memory_store is not None:
            ctx.setdefault("memory_store", memory_store)
        if semantic_memory is not None:
            ctx.setdefault("semantic_memory", semantic_memory)
        if vector_memory is not None:
            ctx.setdefault("vector_memory", vector_memory)
        if "memory" not in ctx:
            ctx["memory"] = memory_facade or memory_store or semantic_memory or vector_memory

        if not ctx.get("objective") and ctx.get("message"):
            ctx["objective"] = ctx["message"]
        elif not ctx.get("message") and ctx.get("objective"):
            ctx["message"] = ctx["objective"]
        return ctx

    @staticmethod
    def _looks_like_unbounded_compute_request(params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        ctx = context or {}
        declared = str(ctx.get("resource_intensity", "") or params.get("resource_intensity", "")).strip().lower()
        if declared in {"unbounded", "extreme", "max", "stress"}:
            return True

        text_parts = [
            str(ctx.get("objective", "") or ""),
            str(ctx.get("message", "") or ""),
            str(params.get("command", "") or ""),
            str(params.get("script", "") or ""),
            str(params.get("query", "") or ""),
        ]
        text = " ".join(part for part in text_parts if part).lower()
        risk_markers = (
            "100 million digits",
            "infinite loop",
            "run forever",
            "max out",
            "stress test",
            "thrash cpu",
            "thrash memory",
            "use all cpu",
            "use all ram",
            "use all memory",
            "use all gpu",
            "use all vram",
        )
        return any(marker in text for marker in risk_markers)

    # Skill name aliases — maps legacy/alternate names to actual registered skill names
    SKILL_ALIASES: Dict[str, str] = {
        "search_web": "web_search",
        "generate_image": "sovereign_imagination",
        "free_search": "web_search",
    }

    @staticmethod
    def _normalize_execution_params(params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(params or {})
        if "params" in normalized and isinstance(normalized["params"], dict):
            nested_params = dict(normalized["params"])
            for key, value in normalized.items():
                if key != "params":
                    nested_params.setdefault(key, value)
            return nested_params
        return normalized

    async def execute(self, skill_name: str, params: Dict[str, Any],
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Safe execution wrapper with adaptivity, security, and retries."""

        # Resolve skill aliases (e.g., "search_web" → "web_search")
        skill_name = self.SKILL_ALIASES.get(skill_name, skill_name)

        # Sanitize double-nested "params" from LLM hallucinations before execution.
        # Preserve any top-level fields we already inferred instead of discarding them.
        normalized_params = self._normalize_execution_params(params)
        if normalized_params != params and "params" in params and isinstance(params["params"], dict):
            self.logger.warning("[%s] Unpacking double-nested params from LLM hallucination.", skill_name)
        params = normalized_params

        constitution = None
        tool_handle = None
        result: Optional[Dict[str, Any]] = None

        @self.error_boundary
        async def _execute_wrapped():
            nonlocal constitution, tool_handle, result
            start_time = time.monotonic()
            ctx = self._augment_execution_context(context)
            exec_source = self._resolve_execution_source(ctx)
            
            # 1. Verification
            if skill_name not in self.skills:
                # ── Pillar 2: Hephaestus (Autonomous Forge) ──
                hephaestus = optional_service("hephaestus_engine", default=None)
                objective = ctx.get("objective") or ctx.get("message")
                
                if hephaestus and objective:
                    self.logger.info("🔨 Tool '%s' missing. Engaging Hephaestus forge...", skill_name)
                    forge_result = await hephaestus.synthesize_skill(skill_name, objective)
                    if forge_result.get("ok"):
                        # Skill should now be registered via discovery in synthesize_skill
                        if skill_name in self.skills:
                            self.logger.info("✅ Skill '%s' forged successfully.", skill_name)
                        else:
                            return {"ok": False, "error": f"Tool '{skill_name}' forge failed (Not registered)."}
                    else:
                        return {"ok": False, "error": f"Tool '{skill_name}' missing and forge failed: {forge_result.get('error')}"}
                else:
                    return {"ok": False, "error": f"Skill '{skill_name}' not found and forge unavailable."}
            
            meta = self.skills[skill_name]
            is_forged = meta.module_path and "skills/" in meta.module_path
            
            # Lazy loading of skill class
            if meta.skill_class is None and not is_forged:
                try:
                    self.logger.info("🧩 Lazy loading skill: %s", skill_name)
                    module = importlib.import_module(meta.module_path)
                    skill_class = getattr(module, meta.class_name)
                    meta.skill_class = skill_class
                    meta.input_model = getattr(skill_class, "input_model", None)
                    # Initialize instance
                    self.instances[skill_name] = skill_class()
                except Exception as e:
                    self.logger.error("Failed to lazy load %s: %s", skill_name, e)
                    return {"ok": False, "error": f"Failed to load implementation: {e}"}

            ok, errors = meta.requirements.check()
            if not ok:
                return {"ok": False, "error": "Missing dependencies", "details": errors}

            # ── CONSTITUTIONAL CLOSURE: Will + AuthorityGateway gated tools ──
            constitutional_runtime_live = False
            try:
                constitutional_runtime_live = (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
                from core.constitution import get_constitutional_core
                from core.executive.authority_gateway import get_authority_gateway

                constitution = get_constitutional_core(self.orchestrator)
                tool_handle = await constitution.begin_tool_execution(
                    skill_name,
                    params,
                    source=exec_source,
                    objective=str(ctx.get("objective") or ctx.get("message") or ""),
                )
                if not tool_handle.approved:
                    reason = str(getattr(tool_handle.decision, "reason", "blocked"))
                    self.logger.warning("🚫 CapabilityEngine: Tool execution '%s' blocked by Constitution: %s", skill_name, reason)
                    failure_markers = ("gate_failed", "required", "unavailable")
                    status = (
                        "blocked_by_executive_gate_failure"
                        if any(marker in reason for marker in failure_markers)
                        else "blocked_by_executive"
                    )
                    return {"ok": False, "error": f"Executive veto: {reason}", "status": status}

                constraints = dict(getattr(tool_handle, "constraints", {}) or {})
                if constraints:
                    merged_constraints = dict(ctx.get("executive_constraints", {}) or {})
                    merged_constraints.update(constraints)
                    ctx["executive_constraints"] = merged_constraints

                capability_token_id = getattr(tool_handle, "capability_token_id", None)
                if constitutional_runtime_live:
                    if not capability_token_id:
                        return {
                            "ok": False,
                            "error": "Capability token missing",
                            "status": "blocked_by_missing_capability_token",
                        }
                    if not get_authority_gateway().verify_tool_access(skill_name, capability_token_id):
                        return {
                            "ok": False,
                            "error": "Capability token denied tool execution",
                            "status": "blocked_by_capability_token",
                        }
                if capability_token_id:
                    ctx["capability_token_id"] = capability_token_id
            except Exception as e:
                if constitutional_runtime_live:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "capability_engine",
                            "constitutional_gate_failed",
                            detail=skill_name,
                            severity="warning",
                            classification="background_degraded",
                            context={"error": type(e).__name__},
                            exc=e,
                        )
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    self.logger.warning("🚫 CapabilityEngine: Executive check failed for '%s': %s", skill_name, e)
                    return {
                        "ok": False,
                        "error": "Constitutional gate unavailable",
                        "status": "blocked_by_executive_gate_failure",
                    }
                self.logger.debug("CapabilityEngine: constitutional check failed, proceeding degraded: %s", e)

            # 2a. Metabolic self-preservation guard
            try:
                metabolism = resolve_metabolic_monitor(default=None)
                repo = resolve_state_repository(default=None)
                current_state = getattr(repo, "_current", None) if repo is not None else None
                phi = float(getattr(current_state, "phi", 0.0) or 0.0) if current_state is not None else 0.0
                snapshot = metabolism.get_current_metabolism() if metabolism else None
                health_score = float(getattr(snapshot, "health_score", 1.0) or 1.0) if snapshot else 1.0
                cpu_percent = float(getattr(snapshot, "cpu_percent", 0.0) or 0.0) if snapshot else 0.0
                ram_percent = float(getattr(snapshot, "ram_percent", 0.0) or 0.0) if snapshot else 0.0
                unbounded = self._looks_like_unbounded_compute_request(params, ctx)
                should_block = False
                reason = ""
                if not meta.is_core_personality:
                    if health_score <= 0.25 and meta.metabolic_cost >= 2:
                        should_block = True
                        reason = f"metabolic_health_critical:{health_score:.2f}"
                    elif health_score <= 0.40 and meta.metabolic_cost >= 3:
                        should_block = True
                        reason = f"metabolic_health_low:{health_score:.2f}"
                    elif unbounded and (health_score <= 0.55 or cpu_percent >= 80.0 or ram_percent >= 85.0):
                        should_block = True
                        reason = (
                            f"substrate_risk:health={health_score:.2f}:"
                            f"cpu={cpu_percent:.1f}:ram={ram_percent:.1f}"
                        )
                    elif phi and phi < 0.18 and meta.metabolic_cost >= 2:
                        should_block = True
                        reason = f"phi_fragility:{phi:.3f}"
                if should_block:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "capability_engine",
                            "metabolic_self_preservation_block",
                            detail=skill_name,
                            severity="warning",
                            classification="background_degraded",
                            context={
                                "reason": reason,
                                "metabolic_cost": getattr(meta, "metabolic_cost", None),
                                "unbounded": unbounded,
                            },
                        )
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    return {
                        "ok": False,
                        "error": f"Self-preservation block: {reason}",
                        "status": "blocked_by_self_preservation",
                    }
            except Exception as e:
                self.logger.debug("CapabilityEngine: metabolic self-preservation check skipped: %s", e)

            # 2. EDI Autonomy & Security Check (Phase 23.4)
            edi = resolve_edi(default=None)
            if edi and hasattr(edi, "can_do"):
                # Infer risk level: > 2 cost is high risk, system mutations are critical
                risk = "low"
                if meta.metabolic_cost >= 3: risk = "high"
                if skill_name in ["run_bash_command", "self_modify", "manage_abilities"]: risk = "critical"
                
                allowed, reason = edi.can_do(skill_name, risk_level=risk)
                if not allowed:
                    self.logger.warning("🛡️ EDI blocked execution of '%s': %s", skill_name, reason)
                    return {"ok": False, "error": f"EDI Security Block: {reason}", "status": "blocked_by_edi"}

            # 3. Adaptation & Security (Rosetta Stone / Sandbox)
            exec_params = params
            if is_forged and self.sandbox:
                self.logger.info("🛡️ Executing FORGED skill '%s' in Sandbox 2.0", skill_name)
                try:
                    code = Path(meta.module_path).read_text()
                    # Run in executor to be non-blocking
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, 
                        lambda: self.sandbox.execute(code, meta.class_name, exec_params)
                    )
                    return result if isinstance(result, dict) else {"ok": True, "result": result}
                except Exception as e:
                    self.logger.error("Sandbox execution failed for %s: %s", skill_name, e)
                    return {"ok": False, "error": f"Sandbox failed: {e}"}

            if self.rosetta_stone:
                params_or_error = self._apply_security(skill_name, exec_params)
                if isinstance(params_or_error, dict) and not params_or_error.get("ok", True): 
                    return params_or_error
                exec_params = params_or_error

            # 3. Instance Management
            if skill_name not in self.instances:
                self.instances[skill_name] = meta.skill_class()
            
            # 4. Critical Execution loop
            self._emit_skill_status(skill_name, "RUNNING")
            
            # 2026 Transcendence: Memory Budget Enforcement
            from core.runtime import CoreRuntime
            try:
                rt = CoreRuntime.get_sync()
                gov = rt.container.get("memory_governor")
                if gov: gov.check()
                orm = rt.container.get("persistent_state")
            except Exception: 
                rt = None
                orm = None

            # --- Central Resilience Primitives: The Cognitive Governor ---
            # Instantiate on the engine if it doesn't exist yet
            if not hasattr(self, "_cognitive_governor"):
                from core.resilience.cognitive_governor import CognitiveGovernor
                self._cognitive_governor = CognitiveGovernor(max_concurrent_tasks=5, base_backoff=1.0)

            try:
                # Execute safely via the Governor to prevent cascading API failures
                async def resilient_call():
                    return await self._execute_with_retry(self.instances[skill_name], skill_name, exec_params, ctx)

                if tool_handle is not None:
                    from core.governance_context import governed_scope

                    async with governed_scope(tool_handle.decision):
                        result = await self._cognitive_governor.execute_safely(
                            task_name=skill_name,
                            coroutine=resilient_call
                        )
                else:
                    result = await self._cognitive_governor.execute_safely(
                        task_name=skill_name,
                        coroutine=resilient_call
                    )
                
            except Exception as e:
                self.logger.error("❌ Skill '%s' unwrapped failure: %s", skill_name, e)
                result = {"ok": False, "error": str(e), "_exception": True}
            
            duration_ms = (time.monotonic() - start_time) * 1000
            
            # Update state based on result
            if result is None:
                result = {"ok": False, "error": "Unknown execution failure (result is None)"}

            # A graceful {ok: false} return means the skill itself is healthy —
            # only mark ERROR if the skill threw an unhandled exception (caught above).
            # This prevents "nmap not installed" from permanently bricking sovereign_network.
            was_exception = result.pop("_exception", False) if isinstance(result, dict) else False
            final_state = "ERROR" if was_exception else "READY"
            self._emit_skill_status(skill_name, final_state)
            if not result.get("ok", True):
                # Store error for diagnostics, but ONLY if the skill is in ERROR state.
                # Graceful {ok: false} (e.g. "nmap not installed") should NOT persist
                # as degraded_reason — the skill is still healthy, just this call failed.
                if was_exception:
                    self.skill_last_errors[skill_name] = str(result.get("error") or "execution_failed")
                # else: transient failure, don't pollute the catalog
            else:
                self.skill_last_errors.pop(skill_name, None)

            # 5. Persistent Audit (ORM)
            if orm:
                try:
                    # Redact sensitive parameters for ORM logging
                    safe_params = params.copy()
                    sensitive_keys = {"password", "token", "api_key", "secret", "credentials", "auth"}
                    for k in safe_params:
                        if any(s in k.lower() for s in sensitive_keys):
                            safe_params[k] = "[REDACTED]"
                            
                    orm.log_execution(
                        skill_name=skill_name,
                        params=safe_params,
                        status=final_state,
                        duration_ms=duration_ms,
                        result=result if result.get("ok") else None,
                        error=result.get("error") if not result.get("ok") else None
                    )
                except Exception as e:
                    self.logger.warning("ORM logging failed: %s", e)
            
            # 5. Mycelium Reinforcement (Sentient Feedback Loop)
            try:
                if hasattr(self.orchestrator, 'mycelium') and self.orchestrator.mycelium:
                    # Issue 53 Fix: Only catch expected reinforcement errors
                    self.orchestrator.mycelium.reinforce(
                        f"skill_{skill_name}", 
                        success=result.get("ok", False)
                    )
            except AttributeError as e:
                self.logger.debug("Reinforcement attribute missing: %s", e)
            except Exception as e:
                self.logger.warning("Reinforcement failed: %s", e)
            
            # 6. Outcome Recording (Asynchronous)
            if self.temporal:
                t = asyncio.create_task(self._record_temporal(skill_name, params, ctx, result))
                t.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
            
            return result
        try:
            return await _execute_wrapped()
        finally:
            try:
                if constitution is not None and tool_handle is not None and bool(getattr(tool_handle, "approved", False)):
                    await constitution.finish_tool_execution(
                        tool_handle,
                        result=result or {"ok": False, "error": "execution_not_completed"},
                        success=bool(isinstance(result, dict) and result.get("ok", False)),
                        duration_ms=0.0,
                        error="" if bool(isinstance(result, dict) and result.get("ok", False)) else str((result or {}).get("error", "")),
                    )
            except Exception as _exc:
                self.logger.debug("Suppressed Exception: %s", _exc)

    def _apply_security(self, skill_name: str, params: Dict[str, Any]) -> Union[Dict[str, Any], Dict[str, str]]:
        """Issue 54: Scoped security adaptation for skill parameters."""
        if not self.rosetta_stone:
            return params
            
        # Issue 54: Only check keys in COMMAND_PARAM_KEYS to avoid security false positives
        COMMAND_PARAM_KEYS = {"command", "cmd", "path", "url", "target", "script"}
        
        def scan_recursive(val: Any, key: Optional[str] = None) -> Tuple[bool, Any, Optional[str]]:
            if isinstance(val, str):
                # Issue 54: Limit security scanning to relevant parameter names
                if key and key.lower() not in COMMAND_PARAM_KEYS:
                    return True, val, None
                    
                # Check for common shell injection patterns
                if any(x in val for x in [";", "&&", "||", "`", "$(", "|", ">", "<"]):
                    threats = self.rosetta_stone.analyze_threat(val)
                    if not threats["safe"]:
                        return False, val, f"Security Block (Threat Detected): {threats['threats']}"
                
                return True, self.rosetta_stone.adapt_command(val), None
            elif isinstance(val, dict):
                new_dict = {}
                for k, v in val.items():
                    ok, new_v, err = scan_recursive(v, k)
                    if not ok: return False, None, err
                    new_dict[k] = new_v
                return True, new_dict, None
            elif isinstance(val, list):
                new_list = []
                for item in val:
                    ok, new_item, err = scan_recursive(item, key)
                    if not ok: return False, None, err
                    new_list.append(new_item)
                return True, new_list, None
            return True, val, None

        ok, filtered_params, error_msg = scan_recursive(params)
        if not ok:
            self.logger.warning("❌ Security violation blocked in skill '%s': %s", skill_name, error_msg)
            return {"ok": False, "error": error_msg, "status": "blocked"}
        
        return filtered_params

    async def _execute_with_retry(self, skill: Any, skill_name: str, params: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a skill method with a retry loop for transient failures."""
        last_error = "Unknown"
        attempt = 0
        for attempt in range(self.max_retries):
            try:
                if attempt > 0: 
                    await asyncio.sleep(self.retry_delay * attempt)
                    self.logger.info("Retrying %s (attempt %s)...", skill_name, attempt+1)

                if hasattr(skill, "safe_execute") and callable(getattr(skill, "safe_execute")):
                    output = await asyncio.wait_for(skill.safe_execute(params, context), timeout=self.timeout)
                else:
                    inputs = self._prepare_inputs(skill, params, context)
                    output = await self._call_method(skill, inputs)
                
                if self._check_success(output):
                    if isinstance(output, dict):
                        payload = dict(output)
                        payload.setdefault("ok", True)
                        payload["retries"] = attempt
                        return payload
                    return {"ok": True, "result": output, "retries": attempt}
                
                last_error = self._extract_error(output)
                if not self._is_transient(last_error): 
                    break
            except Exception as e:
                last_error = str(e)
                if not self._is_transient(last_error): 
                    break
        
        return {"ok": False, "error": last_error, "retries": attempt}

    async def _call_method(self, skill: Any, inputs: Dict[str, Any]) -> Any:
        """Calls the skill method, handling both sync and async."""
        # Phase 4 Sandbox check: If this is an external/forged skill, use Sandbox2
        is_core = getattr(skill, "is_core_personality", False)
        
        # If the skill is not core and we have source code (forged), we should sandbox it.
        # For simplicity, we assume skills loaded from skilled_dir aren't core.
        
        method = skill.execute if hasattr(skill, "execute") else skill
        if inspect.iscoroutinefunction(method):
            return await asyncio.wait_for(method(**inputs), timeout=self.timeout)
        
        # If RestrictedPython is available and NOT core, we could potentially wrap it,
        # but for now we focus on FORGED skills which provide source.
        return await asyncio.get_running_loop().run_in_executor(None, lambda: method(**inputs))

    def _prepare_inputs(self, skill: Any, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Maps parameters to the skill's expected signature."""
        method = skill.execute if hasattr(skill, "execute") else skill
        sig = inspect.signature(method)
        if "goal" in sig.parameters:
            goal_payload: Dict[str, Any]
            if isinstance(params, dict):
                goal_payload = dict(params)
                nested_params = dict(goal_payload.get("params") or {}) if isinstance(goal_payload.get("params"), dict) else {}
                for key, value in goal_payload.items():
                    if key != "params":
                        nested_params.setdefault(key, value)
                goal_payload["params"] = nested_params
            else:
                goal_payload = {"params": {"value": params}}

            objective = (
                goal_payload.get("objective")
                or context.get("objective")
                or context.get("message")
                or goal_payload.get("query")
                or goal_payload.get("content")
                or goal_payload.get("text")
                or goal_payload.get("command")
                or goal_payload.get("path")
            )
            if objective:
                goal_payload["objective"] = str(objective)
            return {"goal": goal_payload, "context": context}
        if "params" in sig.parameters: 
            return {"params": params, "context": context}
        return params

    def _check_success(self, out: Any) -> bool:
        """Determines if the skill output indicates success."""
        if isinstance(out, dict):
            return out.get("ok", True)
        return out is not None

    def _extract_error(self, out: Any) -> str:
        """Extracts an error message from skill output."""
        if isinstance(out, dict):
            return out.get("error") or out.get("message") or "Failed"
        return "Error"

    def _is_transient(self, err: str) -> bool:
        """Checks if an error is likely transient (network, timeout, etc)."""
        return any(x in str(err).lower() for x in ["timeout", "network", "retry", "limit"])

    async def _record_temporal(self, action: str, params: Dict[str, Any], context: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Records the skill outcome to the Temporal Learning system."""
        try:
            await self.temporal.record_outcome(
                action=action, 
                context=str(context)[:200],
                intended_outcome=str(params)[:200],
                actual_outcome=str(result)[:500],
                success=result.get("ok", False)
            )
        except Exception as e:
            self.logger.debug("Temporal record failed: %s", e)

    def get_health(self) -> Dict[str, Any]:
        """Provides extended health data for the capability system."""
        report = super().get_health()
        report["skills_total"] = len(self.skills)
        # Deep check: how many skills have dependencies met
        report["skills_ready"] = len([s for s in self.skills.values() if s.requirements.check()[0]])
        return report
