import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from core.orchestrator.main import RobustOrchestrator
from core.resilience.startup_validator import StartupValidator
from core.container import ServiceContainer

# Configure logging to be less chatty
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Aura.Verify")

async def verify():
    print("\n🛡️ [VERIFICATION] Starting Focused Boot Validation...")
    orch = RobustOrchestrator()
    
    # 1. Manually trigger necessary initializations without full startup
    print("📦 Registering Core Services...")
    
    # Mocking Kernel Interface for validation
    from unittest.mock import MagicMock
    orch.kernel_interface = MagicMock()
    orch.kernel_interface.is_ready.return_value = True
    
    brain_organ = MagicMock()
    brain_organ.instance = MagicMock()
    orch.kernel_interface.kernel.organs = {"brain": brain_organ}
    orch.kernel_interface.kernel.state.version = 5000
    
    # Register required services
    from core.safety.self_preservation_safe import SafeBackupSystem
    from core.resilience.stability_guardian import StabilityGuardian
    from core.state.state_repository import StateRepository
    
    # Ensure they are in container
    ServiceContainer.register_instance("backup_system", SafeBackupSystem())
    ServiceContainer.register_instance("stability_guardian", StabilityGuardian(orch))
    
    # Use a dummy repo or ensure the real one has state
    repo = StateRepository(is_vault_owner=True)
    await repo.initialize() # This sets up SHM and genesis state
    ServiceContainer.register_instance("state_repository", repo)
    
    print("🧐 Running StartupValidator Checks...")
    validator = StartupValidator(orch)
    
    # We want to see the report
    success = await validator.validate_all()
    
    print("\n" + "="*60)
    print(" VERIFICATION RESULTS")
    print("="*60)
    for check in validator.checks:
        status = "PASSED" if check.passed else "FAILED"
        print(f"[{status}] {check.id}: {check.name} - {check.message}")
    
    if success:
        print("\n✅ SYSTEM STABILIZATION CONFIRMED.")
        sys.exit(0)
    else:
        print("\n❌ SYSTEM STABILIZATION FAILED. Remaining issues found.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify())
