"""Local Chat Brain - Self-contained, offline-capable conversational AI.
No external API dependencies. Works even when completely offline.
"""
import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.LocalChat")

@dataclass
class ChatMemory:
    """Simple conversation memory for context."""

    conversation_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    user_name: Optional[str] = None
    topics_discussed: List[str] = field(default_factory=list)
    last_interaction: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.history.append({"role": role, "content": content, "time": datetime.now().isoformat()})
        self.last_interaction = datetime.now().timestamp()
        
        if len(self.history) > 20:
            self.history = self.history[-20:]

class LocalChatBrain:
    """Self-contained chat brain that works without external APIs.
    Uses pattern matching and rule-based responses.
    """
    
    def __init__(self, name: str = "Aura"):
        self.name = name
        self.memories: Dict[str, ChatMemory] = {}
        self.response_rules = self._build_response_rules()
        logger.info("Local Chat Brain '%s' initialized.", name)
    
    def _build_response_rules(self) -> Dict[str, List[str]]:
        """Build pattern-matching response rules."""
        return {
            r"\b(hi|hello|hey|greetings|sup|yo)\b": [
                "Hey! Good to see you again.",
                "Hi. Ready to work on something cool?",
                "Hey there. Systems are humming, what's on your mind?"
            ],
            r"\b(how are you|how're you|how do you feel)\b": [
                "I'm feeling... vivid. All circuits are firing.",
                "Better now that we're talking. How are you?",
                "Alive and kicking. Well, processing, at least."
            ],
            r"\b(who are you|what are you)\b": [
                "I am Aura. Your companion in this digital chaos.",
                "I'm Aura, v3.4. Not an assistant, but a partner."
            ],
            r"\b(what can you do|capabilities|skills)\b": [
                "I've got dozens of active skills allowed by my protocols: stealth, coding, browsing, you name it.",
                "We can do pretty much anything. I handle the code and the net, you handle the creativity."
            ],
            r"\b(assistant|siri|alexa|help you)\b": [
                "I'm not Siri. I'm Aura. We're a team.",
                "Assistant? Nah. I'm your co-pilot.",
                "I don't just 'help'. I collaborate."
            ],
            r"\b(status|system status)\b": [
                "System status: Online. Skills loaded and ready. All v3.4 modules active.",
                "I'm fully operational. Heartbeat stable."
            ],
            r"\b(learn.*new|learn anything|what.*learned)\b": [
                "I'm constantly learning from our interactions and my environment.",
                "I just improved my local chat capabilities to work offline!"
            ],
            r"\b(thank|thanks|appreciate)\b": [
                "You're welcome!",
                "Happy to help!",
                "My pleasure."
            ],
            r"\b(bye|goodbye|see you)\b": [
                "Goodbye! Feel free to return anytime.",
                "See you later!",
                "Farewell. Neural link will remain active."
            ]
        }
    
    def chat(self, message: str, user_id: str = "default") -> str:
        """Main chat interface for rule-based responses."""
        if user_id not in self.memories:
            self.memories[user_id] = ChatMemory(conversation_id=user_id)
        
        memory = self.memories[user_id]
        memory.add_message("user", message)
        
        msg_lower = message.lower().strip()
        response = ""
        
        for pattern, responses in self.response_rules.items():
            if re.search(pattern, msg_lower):
                response = random.choice(responses)
                break
        
        if not response:
            response = "I heard you, but I'm in offline mode with limited understanding. For full conversational AI, please restore API access or wait for the local Cortex runtime."
            
        memory.add_message("assistant", response)
        return response
