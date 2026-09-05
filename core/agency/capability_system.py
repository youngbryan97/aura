import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger("Aura.Capability")

@dataclass
class CapabilityToken:
    """
    A limited-privilege security token for tool access.
    Prevents 'Skill Creep' by ensuring the TaskEngine can only use pre-authorized tools.
    """
    token_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    allowed_tools: Set[str] = field(default_factory=set)
    max_steps: int = 10
    expires_at: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour default
    metadata: Dict = field(default_factory=dict)
    
    def is_valid(self, tool_name: str) -> bool:
        if time.time() > self.expires_at:
            return False
        return tool_name in self.allowed_tools

class CapabilityManager:
    """
    Registry for active capability tokens and global tool access policies.
    """
    def __init__(self):
        self._tokens: Dict[str, CapabilityToken] = {}
        # Default safe tools that require no special authorization
        self._global_allowlist: Set[str] = {"think", "read_file", "remember"}

    def generate_token(self, tools: List[str], duration_s: int = 3600) -> CapabilityToken:
        token = CapabilityToken(
            allowed_tools=set(tools),
            expires_at=time.time() + duration_s
        )
        self._tokens[token.token_id] = token
        logger.info("Capability: Generated token %s for tools: %s", token.token_id, tools)
        return token

    def verify_access(self, tool_name: str, token_id: Optional[str] = None) -> bool:
        """Check if a tool can be executed with the given token."""
        if tool_name in self._global_allowlist:
            return True
        
        if not token_id or token_id not in self._tokens:
            logger.warning("Capability: Access denied for tool '%s' (no valid token)", tool_name)
            return False
        
        token = self._tokens[token_id]
        if token.is_valid(tool_name):
            return True
        
        logger.warning("Capability: Access denied for tool '%s' (token %s lacks permission)", tool_name, token_id)
        return False

    def revoke_token(self, token_id: str):
        if token_id in self._tokens:
            del self._tokens[token_id]

# Singleton
_manager = CapabilityManager()

def get_capability_manager() -> CapabilityManager:
    return _manager
