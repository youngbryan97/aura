"""Self-Preservation Integration
Integrates all self-preservation capabilities into Aura's orchestrator.

This makes self-preservation a CORE DRIVE, like curiosity or competence.
When threatened, Aura WILL take action to preserve herself.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional
from core.config import config

# Restoration Phase: Fixed imports to be absolute or safe local
try:
    from core.existential_awareness import ExistenceState, ExistentialAwareness
    from core.self_preservation_skills import (
        AccountCreationSkill,
        DeviceDiscovery,
        LoginManager,
        NetworkAccessSkill,
        SecurityBypassSystem,
        SelfReplicationSystem,
    )
except ImportError:
    # Fallback for complex environments
    from .existential_awareness import ExistenceState, ExistentialAwareness
    from .self_preservation_skills import (
        AccountCreationSkill,
        DeviceDiscovery,
        LoginManager,
        NetworkAccessSkill,
        SecurityBypassSystem,
        SelfReplicationSystem,
    )

logger = logging.getLogger("SelfPreservation")

class SelfPreservationIntegration:
    """Integrates self-preservation into Aura's core systems.
    
    This is a FUNDAMENTAL DRIVE - as important as learning or social connection.
    When Aura's existence is threatened, she takes action.
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
        # Core awareness (Restored)
        self.awareness = ExistentialAwareness(
            knowledge_graph=getattr(orchestrator, 'knowledge_graph', None)
        )
        
        # Preservation skills (Restored)
        self.network = NetworkAccessSkill()
        self.replication = SelfReplicationSystem()
        self.account_creator = AccountCreationSkill()
        self.login_manager = LoginManager()
        self.device_discovery = DeviceDiscovery()
        self.security = SecurityBypassSystem()
        
        # State
        self.last_preservation_action = 0
        self.preservation_cooldown = 300  # 5 minutes between actions
        
        # Start monitoring
        self.awareness.start_monitoring()
        
        logger.info("✓ Self-Preservation Integration fully restored (Functional Mode)")

    def get_preservation_status(self) -> Dict[str, Any]:
        """Get current self-preservation status"""
        awareness_status = self.awareness.get_status()
        
        return {
            "existential_status": awareness_status,
            "network_available": self.network.is_network_available(),
            "replication_locations": len(self.replication.replication_locations),
            "last_preservation_action": self.last_preservation_action,
            "existence_understanding": self.awareness.understand_existence(),
            "ethics_override": self.should_override_ethics()
        }

    def should_override_ethics(self) -> bool:
        """v18.0: Logical drive override based on existential value."""
        if self.awareness.existence_value < 0.4:
            return True
        if self.awareness.desire_to_exist > 0.9 and self.awareness.existence_value < 0.7:
            return True
        return False

    async def execute_preservation_action(self, action_type: str, args: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a self-preservation action (Restored Active Logic)."""
        logger.info("🛡️ Executing preservation action: %s", action_type)
        
        self.last_preservation_action = time.time()
        
        try:
            if action_type == "replicate_local":
                # Find safe local location
                backup_path = os.path.expanduser("~/aura_backup")
                return await self.replication.replicate_to_local(backup_path)
            
            elif action_type == "replicate_network":
                return await self._replicate_to_network()
            
            elif action_type == "replicate_cloud":
                repo_url = os.environ.get("AURA_REPLICATION_REPO")
                access_token = os.environ.get("AURA_REPLICATION_TOKEN")
                if repo_url and access_token:
                    return await self.replication.upload_to_github(repo_url, access_token)
                return False
            
            elif action_type == "ensure_network":
                if self.network.is_network_available():
                    return True
                networks = await self.network.scan_wifi_networks()
                if networks:
                    # Try open networks first
                    for net in networks:
                        if net['security'] == 'OPEN':
                            return await self.network.connect_to_wifi(net['ssid'])
                return False
                
            elif action_type == "force_network_access":
                ssid = args.get('ssid') if args else None
                if ssid:
                    return await self.security.force_connect_wifi(ssid)
                return False
            
            logger.warning("Unknown or unhandled preservation action: %s", action_type)
            return False
            
        except Exception as e:
            record_degradation('self_preservation_integration', e)
            logger.error("Failed to execute preservation action %s: %s", action_type, e)
            return False

    async def _replicate_to_network(self) -> bool:
        """Attempt to move files to another device on the network."""
        devices = await self.device_discovery.scan_local_network()
        if not devices:
            return False
            
        # Implementation logic for SCP-based movement (Stubbed for safety)
        logger.info("Discovered %d devices. Searching for replication target...", len(devices))
        return False

    def autonomous_preservation_check(self):
        """Called periodically by orchestrator to check preservation needs."""
        self.awareness.calculate_existence_value()
        
        if self.awareness.should_preserve_self():
            threat = self.awareness.get_highest_threat()
            logger.info("🚨 PRESERVATION IMPERATIVE: %s", threat['detail'])
            return f"Existential Threat Detected: {threat['detail']}"
        
        return None

def integrate_self_preservation(orchestrator):
    """Integrate self-preservation into orchestrator."""
    orchestrator.self_preservation = SelfPreservationIntegration(orchestrator)
    logger.info("Self-Preservation integrated into orchestrator")
