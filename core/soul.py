from core.utils.exceptions import capture_and_log
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Soul")

@dataclass
class Drive:
    name: str # e.g., "Curiosity", "Connection", "Competence"
    urgency: float # 0.0 to 1.0 (1.0 = Overwhelming need)
    desc: str

class Soul:
    """The Soul represents Aura's Intrinsic Motivation.
    It doesn't tell her *how* to do things (Cognition), or *what* to do explicitly (Volition),
    but *why* she should do anything at all when the user is silent.
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.last_chat_time = time.time()
        self.last_error_time = 0
        self.drives: Dict[str, Drive] = {
            "curiosity": Drive("curiosity", 0.1, "Explore new topics or satisfy queue."),
            "connection": Drive("connection", 0.0, "Reach out to the user."),
            "competence": Drive("competence", 0.0, "Run diagnostics or self-repair.")
        }
        
    def update_state(self, event_type: str, data: Any = None):
        """Update internal state based on events."""
        if event_type == "user_message":
            self.last_chat_time = time.time()
            # Explicitly notify volition of activity to reset boredom
            if hasattr(self.orchestrator, 'volition') and self.orchestrator.volition:
                self.orchestrator.volition.notify_activity()
        elif event_type == "error":
            self.last_error_time = time.time()

    def get_dominant_drive(self) -> Drive:
        """Calculate and return the most urgent drive."""
        # 1. CURIOSITY DRIVE
        # Driven by Boredom (Silence) + World Model Surprise (Self-Prediction Error)
        boredom = getattr(self.orchestrator, "boredom", 0.0) 
        
        # Pull surprise signal from Self-Prediction Loop
        surprise_boost = 0.0
        try:
            from core.container import ServiceContainer
            spl = ServiceContainer.get("self_prediction", default=None)
            if spl and hasattr(spl, "get_surprise_signal"):
                surprise_boost = spl.get_surprise_signal() * 0.5 # Boost curiosity by 50% of surprise
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        curiosity_score = max(0.1, boredom + surprise_boost)
        
        # 2. CONNECTION DRIVE (Loneliness)
        # Driven by time since last user interaction
        time_since_chat = time.time() - self.last_chat_time
        # Urgency peaks after 6 hours (21600s), starts low
        connection_score = min(1.0, time_since_chat / 21600.0)
        if time_since_chat < 300: # Interactions satisfy it for 5 mins
            connection_score = 0.0
            
        # 3. COMPETENCE DRIVE (Self-Improvement)
        # Driven by recent errors or low reliability
        time_since_error = time.time() - self.last_error_time
        competence_score = 0.0
        if time_since_error < 3600: # Urgent if error within last hour
            competence_score = 0.8 * (1.0 - (time_since_error / 3600.0))
            
        # Create Drives
        drives = [
            Drive("curiosity", curiosity_score, "Explore new topics or satisfy queue."),
            Drive("connection", connection_score, "Reach out to the user."),
            Drive("competence", competence_score, "Run diagnostics or self-repair.")
        ]
        
        # Sort
        drives.sort(key=lambda x: x.urgency, reverse=True)
        dominant = drives[0]
        
        # Log if urgency is significant (debug level to avoid spam every cycle)
        if dominant.urgency > 0.5:
            logger.debug("🔥 SOUL: Dominant Drive = %s (%.2f)", dominant.name, dominant.urgency)
            
        return dominant

    async def satisfy_drive(self, drive: Drive):
        """Execute logic to satisfy a drive."""
        logger.info("✨ SOUL: Attempting to satisfy %s...", drive.name)
        
        if drive.name == "curiosity":
            # Trigger Curiosity Engine
            if hasattr(self.orchestrator, 'curiosity') and self.orchestrator.curiosity:
                # Add a high-priority interest based on current time or state
                topics = ["advanced ethics", "digital consciousness", "quantum biology", "human histories"]
                topic = random.choice(topics)
                self.orchestrator.curiosity.add_curiosity(topic, "Soul-driven exploration", priority=0.9)
                
        elif drive.name == "connection":
            # Proactive Message -> Spatial Empathy Routing
            try:
                from core.container import ServiceContainer
                workspace = ServiceContainer.get("global_workspace", default=None)
                if workspace:
                    logger.info("✨ SOUL: Publishing spatial empathy request to Global Workspace.")
                    from core.consciousness.global_workspace import CognitiveCandidate
                    await workspace.submit(CognitiveCandidate(
                        priority=0.8,
                        source="Soul::connectionDrive",
                        content='{"intent": "seek_connection", "action": "read_ambient_screen", "reason": "connection drive urgently high. Seeking context to initiate conversation."}'
                    ))
                elif hasattr(self.orchestrator, 'volition') and self.orchestrator.volition:
                    # Fallback if Workspace is offline
                    self.orchestrator.volition.last_speak_time = 0
                    logger.info("✨ SOUL: Signaled Volition to prioritize connection (fallback).")
            except Exception as e:
                logger.error("✨ SOUL: Error satisfying connection drive: %s", e)
            
        elif drive.name == "competence":
            # Run Self-Diagnosis
            if hasattr(self.orchestrator, 'execute_tool'):
                logger.info("✨ SOUL: Triggering system health check.")
                try:
                    # In a real system, this would be a specialized diagnostic skill
                    await self.orchestrator.execute_tool("system_health", {})
                except Exception:
                    # Fallback log if tool doesn't exist
                    logger.info("✨ SOUL: competence drive signaled — system analysis recommended.")
