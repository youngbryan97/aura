"""core/neural_feed.py
Phase 17.1: Neural Feed for Strategic progress.
Provides a unified interface for emitting strategic and sensory updates.
"""
import logging
from typing import Any, Dict, Optional
from core.thought_stream import get_emitter
from core.container import ServiceContainer

logger = logging.getLogger("Aura.NeuralFeed")

class NeuralFeed:
    """Standardized event bus for high-level cognitive and strategic pulses."""
    
    def __init__(self):
        self.emitter = get_emitter()
        logger.info("📡 Neural Feed Online")

    def push(self, content: str, title: str = "NEURAL_FEED", category: str = "INFO", **kwargs):
        """Broadcast an event to the feed."""
        # Standardize titles for easier UI filtering
        full_title = f"[{category}] {title}" if category else title
        
        # Emit via ThoughtStream (which bridges to EventBus)
        self.emitter.emit(
            title=full_title,
            content=content,
            level="info",
            category=category,
            **kwargs
        )
        
        # Log for persistence/debug
        logger.info("%s: %s", full_title, content)

    def strategic_update(self, project_name: str, task_desc: str, status: str = "PROGRESS"):
        """Specialized helper for strategic events."""
        self.push(
            content=f"Project '{project_name}': {task_desc}",
            title="STRATEGIC_UPDATE",
            category="STRATEGY",
            status=status
        )

# Singleton registration handled in boot, but helper provided
def get_feed() -> NeuralFeed:
    feed = ServiceContainer.get("neural_feed", default=None)
    if not feed:
        # Emergency init if not in container
        feed = NeuralFeed()
        ServiceContainer.register_instance("neural_feed", feed)
    return feed