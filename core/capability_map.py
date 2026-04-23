"""Capability Map (v3.2)
Cognitive Proprioception - Aura's awareness of her own tools.

This module maps all available capabilities to heuristic triggers,
so Aura can automatically use the right tool without explicit commands.
"""
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.utils.intent_normalization import normalize_memory_intent_text

logger = logging.getLogger("Aura.CapabilityMap")


@dataclass
class Capability:
    """Represents a single capability/tool."""

    name: str
    description: str
    trigger_patterns: list[str]  # Regex patterns that trigger this capability
    handler: Callable[..., Any] | None = None
    is_online: bool = False


class CapabilityMap:
    """Registry of all active tools with heuristic triggers.
    
    This is Aura's "sense" of her own capabilities.
    She can "feel" what tools are available and automatically reach for them.
    """
    
    def __init__(self) -> None:
        self.capabilities: dict[str, Capability] = {}
        self._build_default_capabilities()
        logger.info("✓ Capability Map initialized")
    
    def _build_default_capabilities(self) -> None:
        """Build the default capability map."""
        # Search capability
        self.register(Capability(
            name="web_search",
            description="Search the web for current information",
            trigger_patterns=[
                r"\?$",  # Ends with question mark
                r"what is|what are|who is|who are",
                r"latest|recent|current|today|news",
                r"search for|look up|find out",
                r"weather|stock|price|score",
            ]
        ))
        
        # Memory capability
        self.register(Capability(
            name="memory_recall",
            description="Recall past conversations and stored knowledge",
            trigger_patterns=[
                r"remember|recall|last time|before",
                r"you said|we discussed|earlier",
                r"my |our |history",
                r"did i|have i|did we",
            ]
        ))
        
        # Code execution capability
        self.register(Capability(
            name="code_execute",
            description="Execute Python code for calculations or analysis",
            trigger_patterns=[
                r"calculate|compute|math|equation",
                r"run|execute|code|script",
                r"analyze|parse|process data",
                r"\d+\s*[\+\-\*\/]\s*\d+",  # Math expression
            ]
        ))
        
        # Browser capability
        self.register(Capability(
            name="browser_navigate",
            description="Navigate to and interact with web pages",
            trigger_patterns=[
                r"go to|visit|open|navigate to",
                r"https?://",  # URL
                r"\.com|\.org|\.io|\.net",
                r"website|page|site",
            ]
        ))
        
        # File capability
        self.register(Capability(
            name="file_operations",
            description="Read, write, or modify files",
            trigger_patterns=[
                r"read file|write file|save|load",
                r"create file|delete file|modify",
                r"\.py|\.txt|\.md|\.json|\.yaml",
                r"source code|config|settings",
            ]
        ))
        
        # Self-modification capability
        self.register(Capability(
            name="self_modify",
            description="Modify own code or configuration",
            trigger_patterns=[
                r"improve yourself|upgrade|modify yourself",
                r"your code|your source|autonomy_engine",
                r"fix yourself|optimize yourself",
            ]
        ))

        # === v3.3 New Capabilities ===
        
        # Privacy Capability
        self.register(Capability(
            name="privacy_toggle",
            description="Toggle stealth mode and access VPN functions",
            trigger_patterns=[
                r"stealth mode|go dark|hide yourself",
                r"enable vpn|connect vpn|mask ip",
                r"disable stealth|show yourself",
                r"privacy status|are you hidden",
            ]
        ))
        
        # External Chat Capability
        self.register(Capability(
            name="chat_window",
            description="Open external chat windows",
            trigger_patterns=[
                r"open chat|new window|pop up",
                r"chat window|terminal chat|gui chat",
                r"talk in background|background chat",
            ]
        ))
        
        # Device Discovery Capability
        self.register(Capability(
            name="device_discovery",
            description="Scan network and manage devices",
            trigger_patterns=[
                r"scan network|find devices|who is on wifi",
                r"device scan|network scan|bluetooth scan",
                r"list devices|show network",
            ]
        ))

        # Device Access Capability
        self.register(Capability(
            name="device_connect",
            description="Connect to and access remote devices",
            trigger_patterns=[
                r"connect to|ssh into|remote access",
                r"deploy to|replicate to|move to",
                r"transfer|copy to",
            ]
        ))
        
        # Phantom Browser Capability
        self.register(Capability(
            name="browser_interactive",
            description="Human-like web browsing (visible/hidden)",
            trigger_patterns=[
                r"open browser|show me|watch you browse",
                r"go to website|browse to|navigate to",
                r"search for .* and show me",
                r"hide browser|browse effectively",
            ]
        ))
    
    def register(self, capability: Capability) -> None:
        """Register a capability."""
        self.capabilities[capability.name] = capability
    
    def ping_all(self, registry: Any) -> dict[str, bool]:
        """Ping all capabilities to check if they're online.
        
        Args:
            registry: SkillRegistry to check against
            
        Returns:
            Dict of capability_name -> is_online

        """
        results: dict[str, bool] = {}
        
        for name, cap in self.capabilities.items():
            # Check if skill exists in registry
            skill_exists = False
            if registry and hasattr(registry, 'skills'):
                # Map capability names to skill names
                skill_mappings: dict[str, list[str]] = {
                    "web_search": ["web_search", "search", "duckduckgo"],
                    "memory_recall": ["memory", "recall", "remember", "memory_ops"],
                    "code_execute": ["python", "code", "sandbox", "internal_sandbox", "shell"],
                    "browser_navigate": ["browser", "navigate", "web", "browser_action"],
                    "file_operations": ["file", "read", "write", "file_operation"],
                    "self_modify": ["coding", "self_modification", "self_improvement", "self_evolution"],
                    "chat_window": ["native_chat", "chat", "external_chat"],
                    "device_connect": ["network_ops", "uplink_local", "inter_agent_comm"],
                    "browser_interactive": ["browser_action", "phantom_browser"],
                }
                
                possible_skills = skill_mappings.get(name, [name])
                for skill_name in possible_skills:
                    if skill_name in registry.skills:
                        skill_exists = True
                        break
            
            cap.is_online = skill_exists
            results[name] = skill_exists
        
        online_count = sum(results.values())
        logger.info("Capability ping: %d/%d online", online_count, len(results))
        
        return results
    
    def detect_intent(self, message: str) -> list[str]:
        """Detect which capabilities should be triggered by a message.
        
        Args:
            message: User message
            
        Returns:
            List of capability names to trigger

        """
        triggered: list[str] = []
        message_lower = normalize_memory_intent_text(message)

        for name, cap in self.capabilities.items():
            if not cap.is_online:
                continue
                
            for pattern in cap.trigger_patterns:
                if re.search(pattern, message_lower):
                    triggered.append(name)
                    break  # Only trigger once per capability
        
        if triggered:
            logger.debug("Intent detected: %s for '%s...'", triggered, message[:50])
        
        return triggered
    
    def get_status(self) -> dict[str, Any]:
        """Get capability map status."""
        return {
            "total_capabilities": len(self.capabilities),
            "online": sum(1 for c in self.capabilities.values() if c.is_online),
            "capabilities": {
                name: {"online": cap.is_online, "description": cap.description}
                for name, cap in self.capabilities.items()
            }
        }


# Singleton instance
_capability_map: CapabilityMap | None = None

def get_capability_map() -> CapabilityMap:
    """Get or create the capability map instance."""
    global _capability_map
    if _capability_map is None:
        _capability_map = CapabilityMap()
    return _capability_map
