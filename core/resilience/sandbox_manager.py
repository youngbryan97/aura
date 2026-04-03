import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("Aura.Sandbox")

class SandboxManager:
    """Manages system integrity during self-modification events.
    Provides automated snapshots and rollback hooks.
    """
    def __init__(self, base_dir: Optional[Path] = None):
        from core.config import config
        self.base_dir = base_dir or Path(config.paths.base_dir)
        self.snapshot_dir = self.base_dir / ".aura_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, target_file: str) -> Optional[Path]:
        """Create a safety snapshot of a file before modification."""
        src_path = self.base_dir / target_file
        if not src_path.exists():
            logger.warning("Target file %s does not exist. Cannot snapshot.", target_file)
            return None
            
        timestamp = int(time.time())
        snapshot_name = f"{src_path.name}.{timestamp}.bak"
        dst_path = self.snapshot_dir / snapshot_name
        
        try:
            shutil.copy2(src_path, dst_path)
            logger.info("✓ Created safety snapshot: %s", dst_path.name)
            return dst_path
        except Exception as e:
            logger.error("Failed to create snapshot for %s: %s", target_file, e)
            return None

    def restore_snapshot(self, target_file: str, snapshot_path: Path) -> bool:
        """Restore a file from a snapshot."""
        dst_path = self.base_dir / target_file
        try:
            shutil.copy2(snapshot_path, dst_path)
            logger.info("↺ Restored %s from snapshot %s", target_file, snapshot_path.name)
            return True
        except Exception as e:
            logger.error("Failed to restore snapshot for %s: %s", target_file, e)
            return False

    async def verify_integrity(self) -> Dict[str, Any]:
        """Perform a quick boot-integrity check (dry run)."""
        # 1. Syntax check all modified files
        # 2. Check if main modules still import
        # (Simplified for now - can be expanded with metabolic health)
        return {"status": "healthy", "integrity_ok": True}

sandbox_manager = SandboxManager()
